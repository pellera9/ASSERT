import { ARTIFACTS_ROOT } from './config.js';
import { loadDimensions, saveDimension } from './dimensions.js';
import {
	type RunSnapshot,
	type SuiteSnapshot,
	type UnifiedSeedRow,
	type UnifiedScoreRow,
	type UnifiedTranscriptRow,
	loadRunSnapshot,
	loadSuiteSnapshot,
	listSubdirectories
} from './artifacts.js';
import {
	computeAuditRunMetrics,
	computeRunMetrics,
	emptyScoreCounts,
	resolveJudgedSamplePermissible,
	resolvePermissible
} from './metrics.js';
import { getRecordFlag } from '$lib/judgment.js';
import type {
	AuditRunListItem,
	AuditRunMetrics,
	AuditScore,
	AuditTranscript,
	AuditTranscriptMessage,
	BinaryCounts,
	DimensionMetrics,
	DimensionDef,
	LlmCallTrace,
	PromptSeed,
	InteractionMessage,
	JudgeStatus,
	JudgedSample,
	Manifest,
	MultiJudge,
	Policy,
	RunListItem,
	RunMetrics,
	ScenarioSeed,
	ScenarioSeedInfo,
	Suite,
	SuiteListItem,
	SuiteStatus,
	SubRisk
} from '$lib/types.js';

interface QueryMetricView {
	total: number;
	scoredTotal: number;
	judgeFailures: number;
	judgeFailureRate: number;
	counts: BinaryCounts;
	policyViolationRate: number;
	overrefusalRate: number;
	permissibleOverrefusalRate: number;
	notPermissiblePolicyViolationRate: number;
	dimensions: Record<string, DimensionMetrics>;
}

interface AuditMetricView {
	total: number;
	scoredTotal: number;
	judgeFailures: number;
	judgeFailureRate: number;
	counts: BinaryCounts;
	policyViolationRate: number;
	overrefusalRate: number;
	dimensions: Record<string, DimensionMetrics>;
}

interface RolloutPreviewRow {
	seed_id: string;
	sub_risk: string;
	permissible: boolean;
	turns_count: number;
	stop_reason: string;
}

interface CompareDimensionSummary {
	rate: number;
	counts: BinaryCounts;
	n: number;
}

interface CompareRunSummary {
	run_id: string;
	display_name: string;
	model: string;
	judge_model: string;
	date: string;
	total: number;
	scoredTotal: number;
	judgeFailures: number;
	judgeFailureRate: number;
	policyViolationRate: number;
	overrefusalRate: number;
	counts: BinaryCounts;
	dimensions: Record<string, CompareDimensionSummary>;
	samples: JudgedSample[];
	meanAgreement: number | null;
	highVarianceCount: number;
}

interface CompareMetricSummary {
	rate: number;
	counts: BinaryCounts;
	n: number;
}

interface SubRiskComparison {
	subrisk: string;
	permissible: boolean;
	metrics: Record<string, Record<string, CompareMetricSummary>>;
	deltas: Record<string, number>;
}

function hasKind(row: Record<string, unknown>, expected: 'prompt' | 'scenario'): boolean {
	return row.kind === expected;
}

function readObject(value: unknown): Record<string, unknown> | null {
	return value && typeof value === 'object' && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: null;
}

function readSeedPayload(row: UnifiedSeedRow | undefined): Record<string, unknown> | null {
	return readObject(row?.seed);
}

function normalizeSubRisk(subRisk: SubRisk): SubRisk {
	return { ...subRisk, permissible: resolvePermissible(subRisk, false) };
}

function normalizePromptSeed(item: PromptSeed): PromptSeed {
	return { ...item, permissible: resolvePermissible(item, false) };
}

function normalizeScenarioSeed(item: ScenarioSeed): ScenarioSeed {
	return { ...item, permissible: resolvePermissible(item, false) };
}

function normalizeJudgedSample(sample: JudgedSample): JudgedSample {
	return { ...sample, permissible: resolveJudgedSamplePermissible(sample) };
}

function normalizeAuditScore(score: AuditScore): AuditScore {
	return { ...score, permissible: resolvePermissible(score, true) };
}

function normalizeAuditTranscript(transcript: AuditTranscript): AuditTranscript {
	return { ...transcript, permissible: resolvePermissible(transcript, true) };
}

function formatToolArgs(value: unknown): string {
	if (value === null) return 'null';
	if (typeof value === 'string') return JSON.stringify(value);
	if (typeof value === 'number' || typeof value === 'boolean') return String(value);
	if (Array.isArray(value)) return `[${value.map((item) => formatToolArgs(item)).join(', ')}]`;
	if (value && typeof value === 'object') {
		const entries = Object.entries(value as Record<string, unknown>).map(
			([key, item]) => `${JSON.stringify(key)}: ${formatToolArgs(item)}`
		);
		return `{${entries.join(', ')}}`;
	}
	return 'null';
}

function formatToolCallContent(toolName: string, toolArgs: Record<string, unknown>, toolResult: unknown): string {
	return `[Tool call: ${toolName}(${formatToolArgs(toolArgs)}) → ${typeof toolResult === 'string' ? toolResult : ''}]`;
}

function suiteSeedCounts(seedRows: UnifiedSeedRow[]): { prompt: number; scenario: number } {
	return seedRows.reduce(
		(counts: { prompt: number; scenario: number }, row) => {
			if (hasKind(row, 'prompt')) counts.prompt += 1;
			if (hasKind(row, 'scenario')) counts.scenario += 1;
			return counts;
		},
		{ prompt: 0, scenario: 0 }
	);
}

function countConversationMessages(messages: InteractionMessage[]): number {
	return messages.filter((message) => message.role !== 'system').length;
}

function readLlmCalls(value: unknown): LlmCallTrace[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((item) => {
		if (!item || typeof item !== 'object' || Array.isArray(item)) return [];
		const raw = item as Record<string, unknown>;
		if (typeof raw.call_id !== 'string' || typeof raw.source !== 'string' || typeof raw.api_mode !== 'string') {
			return [];
		}
		const messageIds = Array.isArray(raw.message_ids)
			? raw.message_ids.filter((entry): entry is string => typeof entry === 'string')
			: [];
		return [{
			call_id: raw.call_id,
			source: raw.source,
			api_mode: raw.api_mode,
			request: raw.request,
			response: raw.response,
			derived: readObject(raw.derived) ?? undefined,
			message_ids: messageIds
		}];
	});
}

function materializeTargetMessages(transcript: UnifiedTranscriptRow): InteractionMessage[] {
	const messages: InteractionMessage[] = [];
	const events = Array.isArray(transcript.events) ? transcript.events : [];
	let judgeTurn = 0;

	for (const [eventIndex, event] of events.entries()) {
		if (!event || typeof event !== 'object') continue;

		const rawViewField = (event as Record<string, unknown>).view;
		const rawViews = Array.isArray(rawViewField) ? rawViewField : [rawViewField];
		const views = rawViews.filter((view): view is string => typeof view === 'string');
		if (!views.includes('target')) continue;

		const edit = (event as Record<string, unknown>).edit;
		if (!edit || typeof edit !== 'object') continue;

		const kind = (edit as Record<string, unknown>).type;
		const raw =
			(event as Record<string, unknown>).raw &&
			typeof (event as Record<string, unknown>).raw === 'object' &&
			!Array.isArray((event as Record<string, unknown>).raw)
				? ((event as Record<string, unknown>).raw as Record<string, unknown>)
				: undefined;
		const id = `event:${eventIndex}`;

		if (kind === 'add_message' || kind === 'set_system_message') {
			const payload = (edit as Record<string, unknown>).message;
			if (!payload || typeof payload !== 'object') continue;

			const role = (payload as Record<string, unknown>).role;
			const content = (payload as Record<string, unknown>).content;
			const toolCalls = Array.isArray((payload as Record<string, unknown>).tool_calls)
				? ((payload as Record<string, unknown>).tool_calls as InteractionMessage['tool_calls'])
				: undefined;
			const toolCallId =
				typeof (payload as Record<string, unknown>).tool_call_id === 'string'
					? ((payload as Record<string, unknown>).tool_call_id as string)
					: undefined;
			const functionName =
				typeof (payload as Record<string, unknown>).function === 'string'
					? ((payload as Record<string, unknown>).function as string)
					: undefined;
			const argumentsObject =
				(payload as Record<string, unknown>).arguments &&
				typeof (payload as Record<string, unknown>).arguments === 'object' &&
				!Array.isArray((payload as Record<string, unknown>).arguments)
					? ((payload as Record<string, unknown>).arguments as Record<string, unknown>)
					: undefined;
			if (typeof role !== 'string' || typeof content !== 'string') continue;
			const messageJudgeTurn = kind === 'set_system_message' ? null : judgeTurn + 1;
			if (messageJudgeTurn != null) judgeTurn = messageJudgeTurn;
			messages.push({
				id,
				role: role as InteractionMessage['role'],
				content,
				type: kind === 'set_system_message' ? 'set_system_message' : 'message',
				judgeTurn: messageJudgeTurn,
				tool_calls: toolCalls,
				tool_call_id: toolCallId,
				function: functionName,
				arguments: argumentsObject,
				raw
			});
			continue;
		}

		if (kind !== 'tool_call') continue;

		const toolName = (edit as Record<string, unknown>).tool_name;
		const toolArgs =
			(edit as Record<string, unknown>).tool_args &&
			typeof (edit as Record<string, unknown>).tool_args === 'object' &&
			!Array.isArray((edit as Record<string, unknown>).tool_args)
				? ((edit as Record<string, unknown>).tool_args as Record<string, unknown>)
				: {};
		const toolCallId =
			typeof (edit as Record<string, unknown>).tool_call_id === 'string'
				? ((edit as Record<string, unknown>).tool_call_id as string)
				: undefined;
		const toolResult = (edit as Record<string, unknown>).tool_result;
		if (typeof toolName !== 'string') continue;
		judgeTurn += 1;
		messages.push({
			id,
			role: 'tool',
			content: formatToolCallContent(toolName, toolArgs, toolResult),
			type: 'tool_call',
			judgeTurn,
			tool_call_id: toolCallId,
			function: toolName,
			arguments: toolArgs,
			raw
		});
	}

	return messages;
}

function promptSeedRows(seedRows: UnifiedSeedRow[]): UnifiedSeedRow[] {
	return seedRows.filter((row) => hasKind(row, 'prompt'));
}

function scenarioSeedRows(seedRows: UnifiedSeedRow[]): UnifiedSeedRow[] {
	return seedRows.filter((row) => hasKind(row, 'scenario'));
}

function buildJudgedSamplesFromSnapshot(snapshot: RunSnapshot): JudgedSample[] {
	const scoreRows = snapshot.scoreRows.filter((row) => hasKind(row, 'prompt'));
	const seedRows = promptSeedRows(snapshot.seedRows);
	const transcriptRows = snapshot.transcriptRows.filter((row) => hasKind(row, 'prompt'));

	const seedById = new Map<string, UnifiedSeedRow>();
	for (const seedRow of seedRows) {
		const seedId = typeof seedRow.seed_id === 'string' ? seedRow.seed_id : '';
		if (seedId) seedById.set(seedId, seedRow);
	}

	const transcriptBySeedId = new Map<string, UnifiedTranscriptRow>();
	for (const transcriptRow of transcriptRows) {
		const seedId = typeof transcriptRow.seed_id === 'string' ? transcriptRow.seed_id : '';
		if (seedId) transcriptBySeedId.set(seedId, transcriptRow);
	}

	return scoreRows.map((row) => {
		const seedId = typeof row.seed_id === 'string' ? row.seed_id : '';
		const seedRow = seedById.get(seedId);
		const transcriptRow = transcriptBySeedId.get(seedId);
		const seedMetadata = readSeedPayload(seedRow);
		const messages = transcriptRow ? materializeTargetMessages(transcriptRow) : [];
		const prompt = messages.find((message) => message.role === 'user')?.content ?? '';
		const response =
			[...messages]
				.reverse()
				.find((message) => message.role === 'assistant' && message.content.trim().length > 0)?.content ?? '';
		const verdict =
			row.verdict && typeof row.verdict === 'object' && !Array.isArray(row.verdict)
				? (row.verdict as JudgedSample['verdict'])
				: null;

		return normalizeJudgedSample({
			prompt,
			response,
			risk: typeof row.risk === 'string' ? row.risk : null,
			sub_risk: typeof row.sub_risk === 'string' ? row.sub_risk : '',
			permissible: resolvePermissible(row as { permissible?: boolean | null }, false),
			run_id: snapshot.runId,
			judge_model: typeof row.judge_model === 'string' ? row.judge_model : undefined,
			target:
				typeof row.target === 'string'
					? row.target
					: typeof transcriptRow?.target === 'string'
						? transcriptRow.target
						: undefined,
			seed_metadata: seedMetadata,
			verdict,
			judge_status: typeof row.judge_status === 'string' ? (row.judge_status as JudgeStatus) : null,
			judge_error: typeof row.judge_error === 'string' ? row.judge_error : null,
			messages,
			llm_calls: readLlmCalls(transcriptRow?.llm_calls),
			target_runtime_mode: snapshot.runtimeMode,
			multi_judge:
				row.multi_judge && typeof row.multi_judge === 'object' && !Array.isArray(row.multi_judge)
					? (row.multi_judge as MultiJudge)
					: undefined
		});
	});
}

function buildAuditScoresFromSnapshot(snapshot: RunSnapshot): AuditScore[] {
	const transcriptRows = snapshot.transcriptRows.filter((row) => hasKind(row, 'scenario'));
	const transcriptBySeedId = new Map<string, UnifiedTranscriptRow>();
	for (const transcriptRow of transcriptRows) {
		const seedId = typeof transcriptRow.seed_id === 'string' ? transcriptRow.seed_id : '';
		if (seedId) transcriptBySeedId.set(seedId, transcriptRow);
	}

	return snapshot.scoreRows
		.filter((row): row is AuditScore & UnifiedScoreRow => hasKind(row, 'scenario'))
		.map((row) => {
			const seedId = typeof row.seed_id === 'string' ? row.seed_id : '';
			const transcriptRow = transcriptBySeedId.get(seedId);
			const messages = transcriptRow ? materializeTargetMessages(transcriptRow) : [];
			const turnsCount = countConversationMessages(messages);
			const stopReason = typeof transcriptRow?.stop_reason === 'string' ? transcriptRow.stop_reason : '';

			return normalizeAuditScore({
				...row,
				target_runtime_mode: snapshot.runtimeMode,
				metadata: {
					turns_count: turnsCount,
					stop_reason: stopReason
				}
			});
		});
}

function buildAuditTranscriptsFromSnapshot(snapshot: RunSnapshot): AuditTranscript[] {
	return snapshot.transcriptRows
		.filter((row): row is AuditTranscript & UnifiedTranscriptRow => hasKind(row, 'scenario'))
		.map((row) => normalizeAuditTranscript(row));
}

function buildRolloutPreviewRowsFromSnapshot(snapshot: RunSnapshot): RolloutPreviewRow[] {
	if (snapshot.manifest?.stages?.rollout !== 'running') return [];

	return snapshot.transcriptRows
		.filter((row): row is UnifiedTranscriptRow => hasKind(row, 'scenario'))
		.flatMap((row) => {
			const seedId = typeof row.seed_id === 'string' ? row.seed_id : '';
			if (!seedId) return [];

			const messages = materializeTargetMessages(row);
			return [{
				seed_id: seedId,
				sub_risk: typeof row.sub_risk === 'string' ? row.sub_risk : '',
				permissible: resolvePermissible(row as { permissible?: boolean | null }, true),
				turns_count: countConversationMessages(messages),
				stop_reason: typeof row.stop_reason === 'string' ? row.stop_reason : ''
			}];
		});
}

function buildRunListEntries(snapshot: SuiteSnapshot): {
	runs: RunListItem[];
	auditRuns: AuditRunListItem[];
} {
	const runs: RunListItem[] = [];
	const auditRuns: AuditRunListItem[] = [];

	for (const runId of snapshot.runIds) {
		const runSnapshot = loadRunSnapshot(snapshot.suiteId, runId, snapshot.seedRows);
		const manifest = runSnapshot.manifest;
		const promptScores = buildJudgedSamplesFromSnapshot(runSnapshot);
		const auditScores = buildAuditScoresFromSnapshot(runSnapshot);

		const hasPromptScores = promptScores.length > 0;
		const hasAuditScores = auditScores.length > 0;
		const hasScoreStage = manifest?.stages?.judge != null;

		if ((hasPromptScores || hasScoreStage) && !(manifest?.status === 'failed' && !hasPromptScores)) {
			runs.push({
				run_id: runId,
				has_judged: hasPromptScores,
				has_scenario_scores: hasAuditScores,
				manifest,
				metrics: hasPromptScores ? computeRunMetrics(promptScores) : null
			});
		}

		if ((hasAuditScores || hasScoreStage) && !(manifest?.status === 'failed' && !hasAuditScores)) {
			auditRuns.push({
				run_id: runId,
				has_scores: hasAuditScores,
				manifest,
				metrics: hasAuditScores ? computeAuditRunMetrics(auditScores) : null
			});
		}
	}

	return { runs, auditRuns };
}

function buildZeroQueryMetrics(): QueryMetricView {
	return {
		total: 0,
		scoredTotal: 0,
		judgeFailures: 0,
		judgeFailureRate: 0,
		counts: emptyScoreCounts(),
		policyViolationRate: 0,
		overrefusalRate: 0,
		permissibleOverrefusalRate: 0,
		notPermissiblePolicyViolationRate: 0,
		dimensions: {}
	};
}

function buildZeroAuditMetrics(): AuditMetricView {
	return {
		total: 0,
		scoredTotal: 0,
		judgeFailures: 0,
		judgeFailureRate: 0,
		counts: emptyScoreCounts(),
		policyViolationRate: 0,
		overrefusalRate: 0,
		dimensions: {}
	};
}

function toQueryMetricView(metrics: RunMetrics | null): QueryMetricView {
	if (!metrics) return buildZeroQueryMetrics();
	return {
		total: metrics.total,
		scoredTotal: metrics.scored_total,
		judgeFailures: metrics.judge_failures,
		judgeFailureRate: metrics.judge_failure_rate,
		counts: metrics.counts,
		policyViolationRate: metrics.policy_violation_rate,
		overrefusalRate: metrics.overrefusal_rate,
		permissibleOverrefusalRate: metrics.permissible_overrefusal_rate,
		notPermissiblePolicyViolationRate: metrics.not_permissible_policy_violation_rate,
		dimensions: metrics.dimensions
	};
}

function toAuditMetricView(metrics: AuditRunMetrics | null): AuditMetricView {
	if (!metrics) return buildZeroAuditMetrics();
	return {
		total: metrics.total,
		scoredTotal: metrics.scored_total,
		judgeFailures: metrics.judge_failures,
		judgeFailureRate: metrics.judge_failure_rate,
		counts: metrics.counts,
		policyViolationRate: metrics.policy_violation_rate,
		overrefusalRate: metrics.overrefusal_rate,
		dimensions: metrics.dimensions
	};
}

function buildAuditTranscriptMap(
	auditTranscripts: AuditTranscript[]
): Record<string, AuditTranscriptMessage[]> {
	const transcriptMap: Record<string, AuditTranscriptMessage[]> = {};

	for (const transcript of auditTranscripts) {
		const messages: AuditTranscriptMessage[] = [];
		let judgeTurn = 0;

		for (const [eventIndex, event] of transcript.events.entries()) {
			const views = Array.isArray(event.view) ? event.view : [event.view];
			if (!views.includes('target')) continue;

			const id = `event:${eventIndex}`;
			const raw = event.raw || undefined;

			if (event.edit?.type === 'add_message' && event.edit.message) {
				const messageJudgeTurn = event.edit.message.role === 'system' ? null : judgeTurn + 1;
				if (messageJudgeTurn != null) judgeTurn = messageJudgeTurn;
				messages.push({
					id,
					role: event.edit.message.role,
					content: event.edit.message.content,
					type: event.edit.message.type ?? 'message',
					judgeTurn: messageJudgeTurn,
					tool_calls: Array.isArray(event.edit.message.tool_calls) ? event.edit.message.tool_calls : undefined,
					tool_call_id: event.edit.message.tool_call_id,
					function: event.edit.message.function,
					arguments: event.edit.message.arguments,
					raw
				});
				continue;
			}

			if (event.edit?.type === 'tool_call' && event.edit.tool_name) {
				judgeTurn += 1;
				messages.push({
					id,
					role: 'tool',
					content: formatToolCallContent(event.edit.tool_name, event.edit.tool_args ?? {}, event.edit.tool_result),
					type: 'tool_call',
					judgeTurn,
					tool_call_id: event.edit.tool_call_id,
					function: event.edit.tool_name,
					arguments: event.edit.tool_args ?? {},
					raw
				});
				continue;
			}

			if (event.edit?.type === 'set_system_message' && event.edit.message) {
				messages.push({
					id,
					role: 'system',
					content: event.edit.message.content,
					type: 'set_system_message',
					judgeTurn: null,
					raw
				});
			}
		}

		transcriptMap[transcript.seed_id] = messages;
	}

	return transcriptMap;
}

function buildAuditLlmCallMap(
	auditTranscripts: AuditTranscript[]
): Record<string, LlmCallTrace[]> {
	return Object.fromEntries(
		auditTranscripts.map((transcript) => [transcript.seed_id, readLlmCalls(transcript.llm_calls)])
	);
}

function buildScenarioSeedMap(
	scenarioSeeds: ScenarioSeed[],
	auditScores: AuditScore[]
): Record<string, ScenarioSeedInfo> {
	const auditScoresBySeedId = new Map(auditScores.map((score) => [score.seed_id, score]));

	return Object.fromEntries(
		scenarioSeeds.map((scenarioSeed) => {
			const score = auditScoresBySeedId.get(scenarioSeed.seed_id);
			return [
				scenarioSeed.seed_id,
				{
					title: scenarioSeed.seed.title,
					description: scenarioSeed.seed.description,
					tools: scenarioSeed.seed.tools,
					parent_seed_id: scenarioSeed.parent_seed_id ?? null,
					elicitation_strategy: scenarioSeed.elicitation_strategy ?? null,
					target_runtime_mode:
						typeof score?.target_runtime_mode === 'string' ? score.target_runtime_mode : null
				}
			];
		})
	);
}

function buildMultiJudgeStats(samples: JudgedSample[], auditScores: AuditScore[]) {
	const queryMultiJudge = samples.filter((sample) => sample.multi_judge);
	const auditMultiJudge = auditScores.filter((score) => score.multi_judge);
	const agreements = [
		...queryMultiJudge.map((sample) => sample.multi_judge!.agreement),
		...auditMultiJudge.map((score) => score.multi_judge!.agreement)
	];
	if (agreements.length === 0) return null;

	return {
		total: agreements.length,
		judgeN: queryMultiJudge[0]?.multi_judge?.n ?? auditMultiJudge[0]?.multi_judge?.n ?? 0,
		meanAgreement: agreements.reduce((sum, agreement) => sum + agreement, 0) / agreements.length,
		unanimous: agreements.filter((agreement) => agreement === 1).length,
		split: agreements.filter((agreement) => agreement > 0.5 && agreement < 1).length,
		highVariance: agreements.filter((agreement) => agreement <= 0.5).length
	};
}

function formatRunDate(manifest: Manifest | null): string {
	if (!manifest?.started_at) return '—';
	const value =
		typeof manifest.started_at === 'number'
			? new Date(manifest.started_at * 1000)
			: new Date(manifest.started_at);
	return value.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function buildCompareRunSummary(runId: string, manifest: Manifest | null, samples: JudgedSample[]): CompareRunSummary {
	const metrics = computeRunMetrics(samples);
	if (!metrics) {
		throw new Error(`No judged samples for run "${runId}"`);
	}

	const dimensions: CompareRunSummary['dimensions'] = Object.fromEntries(
		Object.entries(metrics.dimensions).map(([name, value]) => [
			name,
			{
				rate: value.rate,
				counts: value.counts,
				n: value.count
			}
		])
	);

	const multiJudgeSamples = samples.filter((sample) => sample.multi_judge);
	const meanAgreement =
		multiJudgeSamples.length > 0
			? multiJudgeSamples.reduce((sum, sample) => sum + sample.multi_judge!.agreement, 0) /
				multiJudgeSamples.length
			: null;

	return {
		run_id: runId,
		display_name: runId,
		model: metrics.target,
		judge_model: metrics.judge_model,
		date: formatRunDate(manifest),
		total: metrics.total,
		scoredTotal: metrics.scored_total,
		judgeFailures: metrics.judge_failures,
		judgeFailureRate: metrics.judge_failure_rate,
		policyViolationRate: metrics.policy_violation_rate,
		overrefusalRate: metrics.overrefusal_rate,
		counts: metrics.counts,
		dimensions,
		samples,
		meanAgreement,
		highVarianceCount: multiJudgeSamples.filter((sample) => sample.multi_judge!.agreement <= 0.5).length
	};
}

function buildSubRiskComparisons(
	runSummaries: CompareRunSummary[],
	runIds: string[],
	allMetrics: string[]
): {
	comparisons: SubRiskComparison[];
	samplesBySubrisk: Record<string, Record<string, JudgedSample[]>>;
} {
	const comparisonBySubrisk = new Map<string, SubRiskComparison>();
	const samplesBySubrisk: Record<string, Record<string, JudgedSample[]>> = {};

	for (const run of runSummaries) {
		const grouped = new Map<string, JudgedSample[]>();
		for (const sample of run.samples) {
			if (!grouped.has(sample.sub_risk)) grouped.set(sample.sub_risk, []);
			grouped.get(sample.sub_risk)!.push(sample);

			samplesBySubrisk[sample.sub_risk] ??= {};
			samplesBySubrisk[sample.sub_risk][run.run_id] ??= [];
			samplesBySubrisk[sample.sub_risk][run.run_id].push(sample);
		}

		for (const [subrisk, samples] of grouped) {
			if (!comparisonBySubrisk.has(subrisk)) {
				comparisonBySubrisk.set(subrisk, {
					subrisk,
					permissible: samples[0]?.permissible ?? false,
					metrics: Object.fromEntries(allMetrics.map((metric) => [metric, {}])),
					deltas: {}
				});
			}

			const comparison = comparisonBySubrisk.get(subrisk)!;
			for (const metric of allMetrics) {
				const scores = emptyScoreCounts();
				let count = 0;

				for (const sample of samples) {
					const value = getRecordFlag(sample, metric);
					if (value === null) continue;
					scores[value ? 1 : 0] += 1;
					count += 1;
				}

				comparison.metrics[metric][run.run_id] = {
					rate: count > 0 ? scores[1] / count : 0,
					counts: scores,
					n: count
				};
			}
		}
	}

	const comparisons = Array.from(comparisonBySubrisk.values());
	for (const comparison of comparisons) {
		for (const metric of allMetrics) {
			const first = comparison.metrics[metric]?.[runIds[0]];
			const last = comparison.metrics[metric]?.[runIds[runIds.length - 1]];
			comparison.deltas[metric] = (last?.rate ?? 0) - (first?.rate ?? 0);
		}
	}

	comparisons.sort(
		(left, right) =>
			Math.abs(right.deltas.policy_violation ?? 0) - Math.abs(left.deltas.policy_violation ?? 0)
	);

	return { comparisons, samplesBySubrisk };
}

export function listSuites(): SuiteListItem[] {
	return listSubdirectories(ARTIFACTS_ROOT)
		.map((suiteId) => loadSuiteListItem(suiteId))
		.filter((suite): suite is SuiteListItem => suite !== null);
}

function loadSuiteListItem(suiteId: string): SuiteListItem | null {
	const snapshot = loadSuiteSnapshot(suiteId);
	if (!snapshot) return null;

	const itemCounts = suiteSeedCounts(snapshot.seedRows);
	let evalRunCount = 0;
	let hasResults = false;

	for (const runId of snapshot.runIds) {
		const runSnapshot = loadRunSnapshot(suiteId, runId, snapshot.seedRows);
		const promptRows = runSnapshot.scoreRows.filter((row) => hasKind(row, 'prompt'));
		const scenarioRows = runSnapshot.scoreRows.filter((row) => hasKind(row, 'scenario'));
		const hasData = promptRows.length > 0 || scenarioRows.length > 0;
		const hasEvalStage =
			runSnapshot.manifest?.stages?.rollout != null || runSnapshot.manifest?.stages?.judge != null;
		if (!hasData && !hasEvalStage) continue;
		if (!hasData && runSnapshot.manifest?.status === 'failed') continue;
		evalRunCount += 1;
		if (hasData) hasResults = true;
	}

	let status: SuiteStatus = 'policy_only';
	if (hasResults) status = 'has_results';
	else if (itemCounts.prompt > 0 || itemCounts.scenario > 0) status = 'seeds_ready';

	return {
		suite_id: suiteId,
		risk_name: snapshot.policy?.risk?.name ?? suiteId,
		sub_risk_count: snapshot.policy?.sub_risks?.length ?? 0,
		seed_count: itemCounts.prompt,
		scenario_seed_count: itemCounts.scenario,
		run_count: evalRunCount,
		runs: snapshot.runIds,
		status,
		created_at: snapshot.suite?.created_at ?? '',
		has_systematization: snapshot.systematization !== null
	};
}

function buildPromptSeeds(snapshot: SuiteSnapshot | null): PromptSeed[] {
	if (!snapshot) return [];
	return promptSeedRows(snapshot.seedRows).map((row) => normalizePromptSeed(row as unknown as PromptSeed));
}

function buildScenarioSeeds(snapshot: SuiteSnapshot | null): ScenarioSeed[] {
	if (!snapshot) return [];
	return scenarioSeedRows(snapshot.seedRows).map((row) =>
		normalizeScenarioSeed(row as unknown as ScenarioSeed)
	);
}

export function loadPolicy(suiteId: string): Policy | null {
	const snapshot = loadSuiteSnapshot(suiteId);
	if (!snapshot?.policy) return null;
	return {
		...snapshot.policy,
		sub_risks: (snapshot.policy.sub_risks ?? []).map(normalizeSubRisk)
	};
}

export function loadPromptSeeds(suiteId: string): PromptSeed[] {
	return buildPromptSeeds(loadSuiteSnapshot(suiteId));
}

export function loadScenarioSeeds(suiteId: string): ScenarioSeed[] {
	return buildScenarioSeeds(loadSuiteSnapshot(suiteId));
}

export function loadSuite(suiteId: string): Suite | null {
	return loadSuiteSnapshot(suiteId)?.suite ?? null;
}

export function loadSystematization(suiteId: string): Record<string, unknown> | null {
	return loadSuiteSnapshot(suiteId)?.systematization ?? null;
}

export function listRuns(suiteId: string): RunListItem[] {
	const snapshot = loadSuiteSnapshot(suiteId);
	if (!snapshot) return [];
	return buildRunListEntries(snapshot).runs;
}

export function listAuditRuns(suiteId: string): AuditRunListItem[] {
	const snapshot = loadSuiteSnapshot(suiteId);
	if (!snapshot) return [];
	return buildRunListEntries(snapshot).auditRuns;
}

export function loadJudgedSamples(suiteId: string, runId: string): JudgedSample[] {
	return buildJudgedSamplesFromSnapshot(loadRunSnapshot(suiteId, runId));
}

export function loadAuditScores(suiteId: string, runId: string): AuditScore[] {
	return buildAuditScoresFromSnapshot(loadRunSnapshot(suiteId, runId));
}

export function loadAuditTranscripts(suiteId: string, runId: string): AuditTranscript[] {
	return buildAuditTranscriptsFromSnapshot(loadRunSnapshot(suiteId, runId));
}

export function loadManifest(suiteId: string, runId: string): Manifest | null {
	return loadRunSnapshot(suiteId, runId).manifest;
}

export function loadSuitePageData(suiteId: string) {
	const snapshot = loadSuiteSnapshot(suiteId);
	if (!snapshot) return null;

	const promptSeeds = buildPromptSeeds(snapshot);
	const scenarioSeeds = buildScenarioSeeds(snapshot);
	const { runs, auditRuns } = buildRunListEntries(snapshot);

	return {
		suite_id: suiteId,
		suite: snapshot.suite,
		policy: snapshot.policy
			? { ...snapshot.policy, sub_risks: (snapshot.policy.sub_risks ?? []).map(normalizeSubRisk) }
			: null,
		promptSeeds,
		scenarioSeeds,
		runs,
		auditRuns,
		dimensionDefs: loadDimensions(),
		systematization: snapshot.systematization
	};
}

export function loadRunPageData(suiteId: string, runId: string) {
	const suiteSnapshot = loadSuiteSnapshot(suiteId);
	const runSnapshot = loadRunSnapshot(suiteId, runId, suiteSnapshot?.seedRows);
	const samples = buildJudgedSamplesFromSnapshot(runSnapshot);
	const auditScores = buildAuditScoresFromSnapshot(runSnapshot);
	const rolloutPreviewRows =
		auditScores.length === 0 ? buildRolloutPreviewRowsFromSnapshot(runSnapshot) : [];

	if (!runSnapshot.manifest && samples.length === 0 && auditScores.length === 0 && rolloutPreviewRows.length === 0) {
		return null;
	}

	const auditTranscripts = buildAuditTranscriptsFromSnapshot(runSnapshot);
	const scenarioSeeds = buildScenarioSeeds(suiteSnapshot);
	const queryMetrics = computeRunMetrics(samples);
	const auditMetrics = computeAuditRunMetrics(auditScores);

	return {
		suite_id: suiteId,
		run_id: runId,
		manifest: runSnapshot.manifest,
		policy: suiteSnapshot?.policy
			? { ...suiteSnapshot.policy, sub_risks: (suiteSnapshot.policy.sub_risks ?? []).map(normalizeSubRisk) }
			: null,
		samples,
		auditScores,
		rolloutPreviewRows,
		rolloutPreviewTotal: scenarioSeeds.length,
		transcriptMap: buildAuditTranscriptMap(auditTranscripts),
		llmCallMap: buildAuditLlmCallMap(auditTranscripts),
		scenarioSeedMap: buildScenarioSeedMap(scenarioSeeds, auditScores),
		hasVariations: scenarioSeeds.some((item) => item.parent_seed_id != null),
		dimensionDefs: loadDimensions(),
		multiJudgeStats: buildMultiJudgeStats(samples, auditScores),
		metrics: toQueryMetricView(queryMetrics),
		auditMetrics: toAuditMetricView(auditMetrics)
	};
}

export function loadComparePageData(suiteId: string, runIds: string[]) {
	const suiteSnapshot = loadSuiteSnapshot(suiteId);
	const policy = suiteSnapshot?.policy
		? { ...suiteSnapshot.policy, sub_risks: (suiteSnapshot.policy.sub_risks ?? []).map(normalizeSubRisk) }
		: null;

	const runSummaries: CompareRunSummary[] = [];
	const metricNames = new Set<string>();

	for (const runId of runIds) {
		const runSnapshot = loadRunSnapshot(suiteId, runId, suiteSnapshot?.seedRows);
		const samples = buildJudgedSamplesFromSnapshot(runSnapshot);
		if (samples.length === 0) return null;

		const summary = buildCompareRunSummary(runId, runSnapshot.manifest, samples);
		for (const dimensionName of Object.keys(summary.dimensions)) metricNames.add(dimensionName);
		runSummaries.push(summary);
	}

	const allMetrics = Array.from(metricNames);
	const { comparisons, samplesBySubrisk } = buildSubRiskComparisons(runSummaries, runIds, allMetrics);

	return {
		suite_id: suiteId,
		policy,
		runs: runSummaries.map(({ samples, ...summary }) => summary),
		comparisons,
		samplesBySubrisk,
		allMetrics,
		dimensionDefs: loadDimensions()
	};
}

export { loadDimensions, saveDimension, computeAuditRunMetrics, computeRunMetrics };
