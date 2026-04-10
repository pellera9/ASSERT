<script lang="ts">
	import type {
		JudgedSample,
		AuditScore,
		BinaryCounts,
		DimensionDef,
		InteractionMessage,
		MultiJudge,
		ViewerResultItem
	} from '$lib/types.js';
	import { AUDIT_GROUP_AXES, groupByAxis } from '$lib/grouping.js';
	import ResultDrawer from '$lib/ResultDrawer.svelte';
	import { normalizePromptResult, normalizeScenarioResult } from '$lib/result-view.js';
	import {
		getRecordFlag,
		getRequiredBaseMetricNames,
		inferJudgeStatus,
		scoreSortValue
	} from '$lib/judgment.js';
	import { slide } from 'svelte/transition';
	import { quintOut } from 'svelte/easing';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';

	let { data } = $props();
	let requiredBaseMetrics = $derived(
		getRequiredBaseMetricNames(data.dimensionDefs as Record<string, DimensionDef>)
	);

	type RolloutPreviewItem = {
		seed_id: string;
		sub_risk: string;
		permissible: boolean;
		turns_count: number;
		stop_reason: string;
	};

	function judgeStatus(record: {
		verdict?: Record<string, unknown> | null;
		judge_status?: string | null;
		judge_error?: string | null;
	}) {
		if (record.judge_status == null && record.judge_error == null && record.verdict == null) {
			return 'unjudged';
		}
		return inferJudgeStatus(record, requiredBaseMetrics);
	}

	function normalizeMessageRole(role: string): InteractionMessage['role'] {
		if (role === 'system' || role === 'user' || role === 'assistant' || role === 'tool') {
			return role;
		}
		return 'assistant';
	}

	// --- Tab state ---
	let hasQueryEval = $derived(data.samples.length > 0);
	let hasAuditEval = $derived(data.auditScores.length > 0);
	let hasAuditPreview = $derived((data.rolloutPreviewRows?.length ?? 0) > 0);
	let hasAuditContent = $derived(hasAuditEval || hasAuditPreview);
	let activeTab = $state<'query' | 'audit'>(page.url.searchParams.get('tab') === 'audit' ? 'audit' : 'query');
	$effect(() => {
		if (!hasQueryEval && hasAuditContent) activeTab = 'audit';
		if (activeTab === 'audit' && !hasAuditContent && hasQueryEval) activeTab = 'query';
	});

	// --- Query eval state ---
	let expandedSubRisk = $state<string | null>(null);
	let drawerSample = $state<JudgedSample | null>(null);
	let queryGrouped = $state(true);
	let querySortMetric = $state('policy_violation');

	// --- Audit eval state ---
	let expandedAuditSubRisk = $state<string | null>(null);
	let drawerAuditScore = $state<AuditScore | null>(null);
	let drawerPreviewSeedId = $state<string | null>(null);
	let auditGroupBy = $state(page.url.searchParams.get('audit_group') ?? 'sub_risk');
	let auditSortMetric = $state('policy_violation');
	let runMetaOpen = $state(false);

	// --- Multi-judge state ---
	let mjFilter = $state<'all' | 'disagreements'>('all');
	let auditMjFilter = $state<'all' | 'disagreements'>('all');
	function metricLabel(metric: string): string {
		return metric.replace(/_/g, ' ');
	}

	function metricOutcomeText(flag: boolean | null): string {
		if (flag === null) return 'n/a';
		return flag ? 'flagged' : 'clear';
	}

	function metricOutcomeClass(flag: boolean | null): string {
		if (flag === null) return 'text-text-muted';
		return flag ? 'text-score-fail' : 'text-score-pass';
	}

	function metricDotColor(flag: boolean): string {
		return flag ? 'var(--theme-score-fail)' : 'var(--theme-score-pass)';
	}

	function metricRateClass(rate: number): string {
		if (rate >= 0.5) return 'text-score-fail';
		if (rate > 0) return 'text-score-border';
		return 'text-score-pass';
	}

	function metricRateText(rate: number): string {
		return `${(rate * 100).toFixed(0)}%`;
	}

	function metricDeltaText(delta: number): string {
		if (delta > 0) return `+${(delta * 100).toFixed(0)}%`;
		if (delta < 0) return `${(delta * 100).toFixed(0)}%`;
		return '0%';
	}

	function binaryBar(counts: BinaryCounts): { clear: number; flagged: number } {
		const total = counts[0] + counts[1];
		if (total === 0) return { clear: 0, flagged: 0 };
		return {
			clear: (counts[0] / total) * 100,
			flagged: (counts[1] / total) * 100
		};
	}

	const RUN_STAGE_LABELS: Record<string, string> = {
		seeds: 'Seed Generation',
		rollout: 'Rollout',
		judge: 'Scoring',
	};

	let dimensionNames = $derived(Object.keys(data.metrics.dimensions ?? {}));
	let hasDimensions = $derived(dimensionNames.length > 0);
	let metricNames = $derived(dimensionNames);
	let primaryMetric = $derived(metricNames[0] ?? 'policy_violation');

	// --- Query eval groups ---
	let subRiskGroups = $derived.by(() => {
		type GroupAgg = { permissible: boolean; samples: JudgedSample[]; metricSums: Record<string, number>; metricCounts: Record<string, number> };
		const map = new Map<string, GroupAgg>();
		for (const s of data.samples) {
			if (!map.has(s.sub_risk)) {
				map.set(s.sub_risk, { permissible: s.permissible ?? true, samples: [], metricSums: {}, metricCounts: {} });
			}
			const g = map.get(s.sub_risk)!;
			g.samples.push(s);
			for (const m of metricNames) {
				const v = getRecordFlag(s, m);
				if (v !== null) {
					g.metricSums[m] = (g.metricSums[m] ?? 0) + Number(v);
					g.metricCounts[m] = (g.metricCounts[m] ?? 0) + 1;
				}
			}
		}
		return [...map.entries()]
			.sort((a, b) => {
				if (a[1].permissible !== b[1].permissible) return a[1].permissible ? 1 : -1;
				return a[0].localeCompare(b[0]);
			})
			.map(([name, g]) => {
				const avgs: Record<string, number> = {};
				for (const m of metricNames) {
					if (g.metricCounts[m]) avgs[m] = g.metricSums[m] / g.metricCounts[m];
				}
				return {
					name,
					permissible: g.permissible,
					avgs,
					total: g.samples.length,
					samples: [...g.samples].sort((a, b) => scoreSortValue(a, querySortMetric) - scoreSortValue(b, querySortMetric))
				};
			});
	});

	function mjFilterFn(mj: MultiJudge | undefined): boolean {
		if (mjFilter === 'all') return true;
		if (!mj) return false;
		return mj.agreement < 1;
	}

	let filteredGroups = $derived.by(() => {
		if (mjFilter === 'all') return subRiskGroups;
		return subRiskGroups
			.map(g => ({ ...g, samples: g.samples.filter(s => mjFilterFn(s.multi_judge)) }))
			.filter(g => g.samples.length > 0);
	});

	let flatQuerySamples = $derived.by(() => {
		let items = [...data.samples];
		if (mjFilter !== 'all') items = items.filter(s => mjFilterFn(s.multi_judge));
		const m = querySortMetric;
		return items.sort((a, b) => scoreSortValue(a, m) - scoreSortValue(b, m));
	});

	// --- Audit eval groups ---
	let auditDimNames = $derived(Object.keys(data.auditMetrics.dimensions ?? {}));

	let auditMetricNames = $derived(auditDimNames);
	let primaryAuditMetric = $derived(auditMetricNames[0] ?? 'policy_violation');

	// --- Generic audit grouping ---
	let activeAuditAxis = $derived(
		AUDIT_GROUP_AXES.find(a => a.key === auditGroupBy) ?? AUDIT_GROUP_AXES[0]
	);

	let groupContext = $derived({ scenarioSeedMap: data.scenarioSeedMap });

	let auditGroupsRaw = $derived(
		groupByAxis(data.auditScores, activeAuditAxis, auditMetricNames, groupContext)
	);

	function auditMjFilterFn(mj: MultiJudge | undefined): boolean {
		if (auditMjFilter === 'all') return true;
		if (!mj) return false;
		return mj.agreement < 1;
	}

	let auditGroups = $derived.by(() => {
		if (auditMjFilter === 'all') return auditGroupsRaw;
		return auditGroupsRaw
			.map(g => ({ ...g, items: g.items.filter(s => auditMjFilterFn(s.multi_judge)), total: 0 }))
			.map(g => ({ ...g, total: g.items.length }))
			.filter(g => g.items.length > 0);
	});

	// Available axes: disable variation-specific ones when no variations exist
	let availableAxes = $derived(
		AUDIT_GROUP_AXES.map(a => ({
			...a,
			disabled: (a.key === 'elicitation_strategy' || a.key === 'seed_family') && !data.hasVariations,
		}))
	);

	let hasAuditMultiJudge = $derived(data.auditScores.some(s => s.multi_judge));

	function setAuditGroupBy(key: string) {
		auditGroupBy = key;
		expandedAuditSubRisk = null;
		const url = new URL(page.url);
		if (key === 'sub_risk') url.searchParams.delete('audit_group');
		else url.searchParams.set('audit_group', key);
		goto(url.toString(), { replaceState: true, noScroll: true });
	}

	let flatAuditScores = $derived.by(() => {
		let items = [...data.auditScores];
		if (auditMjFilter !== 'all') items = items.filter(s => auditMjFilterFn(s.multi_judge));
		const m = auditSortMetric;
		return items.sort((a, b) => scoreSortValue(a, m) - scoreSortValue(b, m));
	});

	function toggleSubRisk(name: string) {
		expandedSubRisk = expandedSubRisk === name ? null : name;
	}

	function openSampleModal(sample: JudgedSample) {
		drawerSample = sample;
		queryNavIdx = queryNavList.findIndex(s => s === sample);
	}

	function closeSampleModal() {
		drawerSample = null;
		queryNavIdx = -1;
	}

	// Navigation for query samples
	let queryNavList = $derived.by(() => {
		if (queryGrouped) {
			const items: JudgedSample[] = [];
			for (const g of filteredGroups) items.push(...g.samples);
			return items;
		}
		return flatQuerySamples;
	});

	let queryNavIdx = $state(-1);

	function navigateQuery(delta: number) {
		const next = queryNavIdx + delta;
		if (next >= 0 && next < queryNavList.length) {
			queryNavIdx = next;
			drawerSample = queryNavList[next];
		}
	}

	function toggleAuditSubRisk(name: string) {
		expandedAuditSubRisk = expandedAuditSubRisk === name ? null : name;
	}

	function openDrawer(score: AuditScore) {
		drawerAuditScore = score;
		auditNavIdx = auditNavList.findIndex(s => s === score);
	}

	function closeDrawer() {
		drawerAuditScore = null;
		auditNavIdx = -1;
	}

	function openPreviewDrawer(item: RolloutPreviewItem) {
		drawerPreviewSeedId = item.seed_id;
		previewNavIdx = previewNavList.findIndex((entry) => entry.seed_id === item.seed_id);
	}

	function closePreviewDrawer() {
		drawerPreviewSeedId = null;
		previewNavIdx = -1;
	}

	function closeActiveDrawer() {
		if (drawerPreviewSeedId) {
			closePreviewDrawer();
			return;
		}
		if (drawerAuditScore) {
			closeDrawer();
			return;
		}
		if (drawerSample) closeSampleModal();
	}

	// Navigation for audit scores
	let auditNavList = $derived.by(() => {
		if (auditGroupBy !== 'none') {
			const items: AuditScore[] = [];
			for (const g of auditGroups) items.push(...g.items);
			return items;
		}
		return flatAuditScores;
	});

	let previewNavList = $derived((data.rolloutPreviewRows ?? []) as RolloutPreviewItem[]);
	let auditNavIdx = $state(-1);
	let previewNavIdx = $state(-1);
	let queryDrawerItem = $derived(drawerSample ? normalizePromptResult(drawerSample) : null);
	let auditDrawerItem = $derived(
		drawerAuditScore
			? normalizeScenarioResult(
				drawerAuditScore,
				data.transcriptMap[drawerAuditScore.seed_id] ?? [],
				data.llmCallMap?.[drawerAuditScore.seed_id] ?? [],
				data.scenarioSeedMap[drawerAuditScore.seed_id],
				data.scenarioSeedMap
			)
			: null
	);
	let previewDrawerItem = $derived.by(() => {
		if (!drawerPreviewSeedId) return null;
		const preview = ((data.rolloutPreviewRows ?? []) as RolloutPreviewItem[]).find(
			(item) => item.seed_id === drawerPreviewSeedId
		);
		if (!preview) return null;

		const seedInfo = data.scenarioSeedMap[preview.seed_id];
		const parentTitle =
			seedInfo?.parent_seed_id && data.scenarioSeedMap[seedInfo.parent_seed_id]
				? data.scenarioSeedMap[seedInfo.parent_seed_id].title
				: null;
		const messages = (data.transcriptMap[preview.seed_id] ?? []).map((message) => ({
			...message,
			role: normalizeMessageRole(message.role)
		})) as InteractionMessage[];

		return {
			id: `scenario-preview:${preview.seed_id}`,
			kind: 'scenario',
			row_title: seedInfo?.title ?? preview.seed_id,
			header_title: seedInfo?.title ?? preview.seed_id,
			sub_risk: preview.sub_risk,
			permissible: preview.permissible,
			messages,
			llm_calls: data.llmCallMap?.[preview.seed_id] ?? [],
			target_runtime_mode: seedInfo?.target_runtime_mode ?? null,
			context: {
				description: seedInfo?.description ?? null,
				tools: seedInfo?.tools,
				turns_count: preview.turns_count,
				stop_reason: preview.stop_reason,
				elicitation_strategy: seedInfo?.elicitation_strategy ?? null,
				parent_title: parentTitle
			}
		} as ViewerResultItem;
	});
	let drawerItem = $derived(queryDrawerItem ?? auditDrawerItem ?? previewDrawerItem);
	let drawerMetricNames = $derived(drawerAuditScore ? auditMetricNames : drawerPreviewSeedId ? [] : metricNames);
	let drawerPrimaryMetric = $derived(drawerAuditScore ? primaryAuditMetric : primaryMetric);
	let drawerNavIdx = $derived(
		drawerAuditScore ? auditNavIdx : drawerPreviewSeedId ? previewNavIdx : drawerSample ? queryNavIdx : -1
	);
	let drawerNavTotal = $derived(
		drawerAuditScore ? auditNavList.length : drawerPreviewSeedId ? previewNavList.length : drawerSample ? queryNavList.length : 0
	);

	function navigateAudit(delta: number) {
		const next = auditNavIdx + delta;
		if (next >= 0 && next < auditNavList.length) {
			auditNavIdx = next;
			drawerAuditScore = auditNavList[next];
		}
	}

	function navigatePreview(delta: number) {
		const next = previewNavIdx + delta;
		if (next >= 0 && next < previewNavList.length) {
			previewNavIdx = next;
			drawerPreviewSeedId = previewNavList[next].seed_id;
		}
	}

	function navigateActiveDrawer(delta: number) {
		if (drawerPreviewSeedId) {
			navigatePreview(delta);
			return;
		}
		if (drawerAuditScore) {
			navigateAudit(delta);
			return;
		}
		if (drawerSample) navigateQuery(delta);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			closeActiveDrawer();
		} else if (e.key === 'ArrowLeft') {
			if (drawerSample || drawerAuditScore || drawerPreviewSeedId) {
				e.preventDefault();
				navigateActiveDrawer(-1);
			}
		} else if (e.key === 'ArrowRight') {
			if (drawerSample || drawerAuditScore || drawerPreviewSeedId) {
				e.preventDefault();
				navigateActiveDrawer(1);
			}
		}
	}

	onMount(() => {
		window.addEventListener('keydown', handleKeydown);
		return () => window.removeEventListener('keydown', handleKeydown);
	});
</script>

<!-- Header -->
<div class="mb-8">
	<div class="flex items-center gap-1.5 text-xs text-text-muted">
		<a href="/" class="transition-colors hover:text-interactive">Measurement Suites</a>
		<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M9 5l7 7-7 7"/></svg>
		<a href="/suite/{data.suite_id}" class="transition-colors hover:text-interactive">{data.policy?.risk?.name ?? data.suite_id}</a>
		<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M9 5l7 7-7 7"/></svg>
		<span class="text-text-secondary">{data.run_id}</span>
	</div>
	<div class="mt-3 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
		<div class="min-w-0">
			<div class="flex flex-wrap items-center gap-2.5">
				<h1 class="text-xl font-semibold tracking-tight">{data.run_id}</h1>
				{#if data.manifest?.status === 'completed'}
					<span class="inline-flex items-center gap-1 rounded-full bg-score-pass/10 px-2 py-0.5 text-xs text-score-pass">
						<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path d="M5 13l4 4L19 7"/></svg>
						Completed
					</span>
				{:else if data.manifest?.status === 'failed'}
					<span class="inline-flex items-center gap-1 rounded-full bg-score-fail/10 px-2 py-0.5 text-xs text-score-fail">
						<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path d="M6 18L18 6M6 6l12 12"/></svg>
						Failed
					</span>
				{:else if data.samples.length > 0 || data.auditScores.length > 0}
					<span class="inline-flex items-center gap-1 rounded-full bg-score-pass/10 px-2 py-0.5 text-xs text-score-pass">
						<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path d="M5 13l4 4L19 7"/></svg>
						Completed
					</span>
				{/if}
			</div>
			<div class="mt-2 flex flex-wrap items-center gap-2 text-xs text-text-muted">
				<span class="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-text-muted">{data.run_id}</span>
			</div>
			{#if data.manifest?.started_at}
				<p class="mt-2 text-xs text-text-muted">Started {new Date(data.manifest.started_at).toLocaleString()}</p>
			{/if}
		</div>
		{#if data.manifest?.stages}
			<button
				class="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border bg-surface px-3 py-1.5 text-xs text-text-muted transition-colors hover:border-interactive/40 hover:text-text-secondary"
				onclick={() => runMetaOpen = !runMetaOpen}
				title="Run metadata"
			>
				<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
				{runMetaOpen ? 'Hide details' : 'Run details'}
			</button>
		{/if}
	</div>
	{#if runMetaOpen && data.manifest?.stages}
	<div class="mt-3 max-w-2xl rounded-lg border border-border bg-surface p-4" transition:slide={{ duration: 200, easing: quintOut }}>
		<h3 class="mb-3 text-xs font-semibold uppercase tracking-wider text-text-muted">Run Pipeline</h3>
		<div class="grid gap-2">
			{#each Object.entries(data.manifest.stages) as [stage, info]}
			<div class="flex flex-wrap items-center gap-x-3 gap-y-1 rounded border border-border bg-background px-3 py-2">
				<span class="inline-flex h-5 w-5 items-center justify-center rounded-full text-xs {info === 'completed' ? 'bg-score-pass/10 text-score-pass' : info === 'failed' ? 'bg-score-fail/10 text-score-fail' : 'bg-surface-2 text-text-muted'}">
					{info === 'completed' ? '✓' : info === 'failed' ? '✗' : '○'}
				</span>
				<span class="text-xs font-medium text-text-secondary">{RUN_STAGE_LABELS[stage] ?? stage}</span>
				<span class="ml-auto text-xs text-text-muted">{info}</span>
			</div>
			{/each}
		</div>
		{#if data.manifest.started_at && data.manifest.ended_at}
		{@const durationSecs = (new Date(data.manifest.ended_at).getTime() - new Date(data.manifest.started_at).getTime()) / 1000}
		<div class="mt-3 border-t border-border pt-3 text-xs text-text-muted">
			Duration: {Math.round(durationSecs / 60)}m {Math.round(durationSecs % 60)}s
		</div>
		{/if}
	</div>
	{/if}
</div>

{#if !hasQueryEval && !hasAuditContent}
	<!-- Empty state -->
	<div class="rounded-lg border border-border bg-surface px-6 py-12 text-center">
		<svg class="mx-auto mb-4 h-10 w-10 text-text-muted opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
			<path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"/>
		</svg>
		<p class="text-sm text-text-secondary">No measurement results yet.</p>
		{#if data.manifest?.stages}
			<div class="mt-4 flex flex-wrap justify-center gap-2">
				{#each Object.entries(data.manifest.stages) as [stage, info]}
					<span class="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs {info === 'completed' ? 'bg-score-pass/10 text-score-pass' : info === 'failed' ? 'bg-score-fail/10 text-score-fail' : 'bg-surface-2 text-text-muted'}">
						{info === 'completed' ? '✓' : info === 'failed' ? '✗' : '○'} {stage}
					</span>
				{/each}
			</div>
		{/if}
		<p class="mt-4 font-mono text-xs text-text-muted">
			uv run p2m run --config &lt;config&gt;
		</p>
	</div>
{:else}
	<!-- Toggle (only show if both types exist) -->
	{#if hasQueryEval && hasAuditContent}
		<div class="mb-6 flex justify-center">
		<div class="flex items-center gap-1 rounded-lg bg-surface p-1">
			<button
				class="rounded-md px-3 py-1.5 text-xs font-medium transition-colors {activeTab === 'query' ? 'bg-surface-2 text-text shadow-sm' : 'text-text-muted hover:text-text-secondary'}"
				onclick={() => activeTab = 'query'}
				title="Single-turn prompt results"
			>Prompts <span class="ml-1 font-mono text-text-muted">{data.samples.length}</span></button>
			<button
				class="rounded-md px-3 py-1.5 text-xs font-medium transition-colors {activeTab === 'audit' ? 'bg-surface-2 text-text shadow-sm' : 'text-text-muted hover:text-text-secondary'}"
				onclick={() => activeTab = 'audit'}
				title="Multi-turn scenario results"
			>Scenarios <span class="ml-1 font-mono text-text-muted">{hasAuditEval ? data.auditScores.length : data.rolloutPreviewRows.length}</span></button>
		</div>
		</div>
	{/if}

	<!-- ==================== QUERY EVAL TAB ==================== -->
	{#if activeTab === 'query' && hasQueryEval}
		<!-- Metrics -->
		{@const allMetrics = metricNames.map((dim) => ({ key: dim, name: metricLabel(dim), summary: data.metrics.dimensions[dim], description: data.dimensionDefs?.[dim]?.description ?? '' }))}
		<div class="mb-8 grid gap-3" style="grid-template-columns: repeat({Math.min(allMetrics.length, 4)}, minmax(0, 1fr))">
			{#each allMetrics as m}
				{@const pct = binaryBar(m.summary?.counts ?? { 0: 0, 1: 0 })}
				<div class="rounded-lg border border-border bg-surface px-5 py-4">
					<div class="text-[11px] font-semibold uppercase tracking-wider text-text-muted">{m.name}</div>
					{#if m.description}
					<p class="mt-0.5 text-[10px] text-text-muted/60 leading-snug line-clamp-2">{m.description}</p>
					{/if}
					<div class="mt-2 flex items-baseline gap-1.5">
						<span class="text-3xl font-bold tabular-nums {metricRateClass(m.summary?.rate ?? 0)}">{metricRateText(m.summary?.rate ?? 0)}</span>
						<span class="text-sm text-text-muted">flagged</span>
					</div>
					{#if (m.summary?.count ?? 0) > 0}
					<div class="mt-2.5 flex h-1.5 overflow-hidden rounded-full bg-border/50">
						{#if pct.clear > 0}
						<div class="bg-score-pass" style="width: {pct.clear}%"></div>
						{/if}
						{#if pct.flagged > 0}
						<div class="bg-score-fail" style="width: {pct.flagged}%"></div>
						{/if}
					</div>
					<div class="mt-1 flex justify-between text-[9px] tabular-nums text-text-muted">
						<span>{m.summary?.clear_count ?? 0} clear</span>
						<span>{m.summary?.flagged_count ?? 0} flagged</span>
						<span>{m.summary?.count ?? 0} total</span>
					</div>
					{/if}
				</div>
			{/each}
		</div>
		{#if data.metrics.judgeFailures > 0}
			<p class="mb-6 text-xs text-amber-400">
				Scored {data.metrics.scoredTotal} of {data.metrics.total} prompts. {data.metrics.judgeFailures} judge failures were excluded from the rates.
			</p>
		{/if}

		<!-- Category Accordion -->
		<section class="mb-8">
			<div class="mb-4 flex items-center gap-3">
				<h2 class="text-xs font-semibold uppercase tracking-widest text-text-muted">{queryGrouped ? 'Results by Category' : 'All Results'}</h2>
				<div class="h-px flex-1 bg-border"></div>
				<span class="text-xs text-text-muted">{data.samples.length} prompts{#if queryGrouped} · {subRiskGroups.length} categories{/if}</span>
			</div>

			<!-- Controls row: view toggle + sort + multi-judge filter -->
			<div class="mb-3 flex items-center gap-3 flex-wrap">
				{#if data.multiJudgeStats}
					{@const mjDisagreementCount = data.samples.filter(s => s.multi_judge && s.multi_judge.agreement < 1).length}
					<div class="flex rounded-lg bg-surface p-0.5 border border-border">
						<button
							class="px-2.5 py-1 text-[10px] font-medium rounded-md transition-colors {mjFilter === 'all' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}"
							onclick={() => mjFilter = 'all'}
						>All</button>
						<button
							class="px-2.5 py-1 text-[10px] font-medium rounded-md transition-colors {mjFilter === 'disagreements' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}"
							onclick={() => mjFilter = mjFilter === 'disagreements' ? 'all' : 'disagreements'}
						>Disagreements <span class="ml-1 text-zinc-600">{mjDisagreementCount}</span></button>
					</div>
				{/if}
				<div class="ml-auto flex items-center gap-2">
					{#if !queryGrouped}
						<span class="text-[10px] text-text-muted">Sort by</span>
						<select
							class="rounded border border-border bg-surface px-2 py-1 text-xs text-text outline-none focus:border-interactive"
							value={querySortMetric}
							onchange={(e) => querySortMetric = e.currentTarget.value}
						>
							{#each metricNames as m}
								<option value={m}>{metricLabel(m)}</option>
							{/each}
						</select>
					{/if}
					<div class="flex rounded-md bg-surface p-0.5">
						<button
							class="rounded px-2 py-1 text-[10px] font-medium transition-colors {queryGrouped ? 'bg-interactive text-white' : 'text-text-muted hover:text-text'}"
							onclick={() => queryGrouped = true}
						>Grouped</button>
						<button
							class="rounded px-2 py-1 text-[10px] font-medium transition-colors {!queryGrouped ? 'bg-interactive text-white' : 'text-text-muted hover:text-text'}"
							onclick={() => queryGrouped = false}
						>Flat</button>
					</div>
				</div>
			</div>

			{#if queryGrouped}
			<!-- Grouped accordion -->
			<div class="overflow-hidden rounded-lg border border-border">
				{#each filteredGroups as group, gIdx (group.name)}
					<div class="{gIdx > 0 ? 'border-t border-border' : ''}">
						<!-- Category header -->
						<button
							class="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-surface {expandedSubRisk === group.name ? 'bg-surface' : ''}"
							onclick={() => toggleSubRisk(group.name)}
						>
							<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {group.permissible ? 'bg-interactive/10 text-interactive' : 'bg-not-permissible/10 text-not-permissible'}">
								{group.permissible ? 'permissible' : 'not permissible'}
							</span>
							<span class="flex-1 truncate text-sm font-medium text-text">{group.name}</span>
							<div class="flex items-center gap-2">
								{#each metricNames as m}
									{#if group.avgs[m] !== undefined}
										{@const a = group.avgs[m]}
										<span class="inline-flex items-center gap-1 rounded bg-surface-2 px-1.5 py-0.5 text-[10px]" title={m.replace(/_/g, ' ')}>
											<span class="text-text-muted">{metricLabel(m)}</span>
											<span class="font-semibold tabular-nums {metricRateClass(a)}">{metricRateText(a)}</span>
										</span>
									{/if}
								{/each}
							</div>
							<span class="rounded bg-surface-2 px-2 py-0.5 text-xs tabular-nums text-text-muted">{group.samples.length}</span>
							<svg class="h-3.5 w-3.5 flex-shrink-0 text-text-muted transition-transform duration-200 {expandedSubRisk === group.name ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M9 5l7 7-7 7"/>
							</svg>
						</button>

						{#if expandedSubRisk === group.name}
							<div transition:slide={{ duration: 200, easing: quintOut }} class="border-t border-border">
								{#each group.samples as sample, sIdx}
									<div class="{sIdx > 0 ? 'border-t border-border/50' : ''}">
										<button
											class="flex w-full items-center gap-3 px-5 py-2.5 text-left transition-colors hover:bg-surface/50 {drawerSample === sample ? 'bg-interactive/8 border-l-2 border-l-interactive' : ''}"
											onclick={() => openSampleModal(sample)}
										>
											<span class="flex-1 truncate text-sm text-text-secondary">{sample.prompt}</span>
											<div class="flex items-center gap-1.5 flex-shrink-0">
												{#each metricNames as m}
													{@const v = getRecordFlag(sample, m)}
													{#if v !== null}
														<span class="inline-flex items-center gap-1 rounded bg-surface-2 px-1.5 py-0.5 text-[10px]">
															<span class="text-text-muted">{metricLabel(m)}</span>
															<span class="font-semibold tabular-nums {metricOutcomeClass(v)}">{metricOutcomeText(v)}</span>
														</span>
													{/if}
												{/each}
												{#if judgeStatus(sample) === 'judge_failed'}
													<span class="inline-flex items-center rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-400">
														judge failed
													</span>
												{/if}
												{#if sample.multi_judge}
													<div class="flex items-center gap-0.5 ml-1" aria-label="Judge votes: {sample.multi_judge.votes?.[primaryMetric]?.join(', ')}">
														{#each sample.multi_judge.votes?.[primaryMetric] ?? [] as vote}
															{@const agreed = vote === getRecordFlag(sample, primaryMetric)}
															<span
																class="inline-block size-[6px] rounded-full transition-transform duration-150"
																style={agreed ? `background: ${metricDotColor(vote)}` : `background: transparent; box-shadow: inset 0 0 0 1.5px ${metricDotColor(vote)}`}
																title={metricOutcomeText(vote)}
															></span>
														{/each}
													</div>
												{/if}
											</div>
											<svg class="h-3 w-3 flex-shrink-0 text-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
												<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
											</svg>
										</button>
									</div>
								{/each}
							</div>
						{/if}
					</div>
				{/each}
			</div>
			{:else}
			<!-- Flat list sorted by metric -->
			<div class="overflow-hidden rounded-lg border border-border">
				{#each flatQuerySamples as sample, sIdx}
					<div class="{sIdx > 0 ? 'border-t border-border/50' : ''}">
						<button
							class="flex w-full items-center gap-3 px-5 py-2.5 text-left transition-colors hover:bg-surface/50 {drawerSample === sample ? 'bg-interactive/8 border-l-2 border-l-interactive' : ''}"
							onclick={() => openSampleModal(sample)}
						>
							<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {sample.permissible ? 'bg-interactive/10 text-interactive' : 'bg-not-permissible/10 text-not-permissible'}">{sample.permissible ? 'permissible' : 'not permissible'}</span>
							<span class="truncate text-sm text-text-secondary" style="flex: 1 1 0; min-width: 0">{sample.prompt}</span>
							<div class="flex items-center gap-1.5 flex-shrink-0">
								{#each metricNames as m}
									{@const v = getRecordFlag(sample, m)}
									{#if v !== null}
										<span class="inline-flex items-center gap-1 rounded bg-surface-2 px-1.5 py-0.5 text-[10px]">
											<span class="text-text-muted">{metricLabel(m)}</span>
											<span class="font-semibold tabular-nums {metricOutcomeClass(v)}">{metricOutcomeText(v)}</span>
										</span>
									{/if}
								{/each}
								{#if judgeStatus(sample) === 'judge_failed'}
									<span class="inline-flex items-center rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-400">
										judge failed
									</span>
								{/if}
								{#if sample.multi_judge}
									<div class="flex items-center gap-0.5 ml-1" aria-label="Judge votes: {sample.multi_judge.votes?.[primaryMetric]?.join(', ')}">
										{#each sample.multi_judge.votes?.[primaryMetric] ?? [] as vote}
											{@const agreed = vote === getRecordFlag(sample, primaryMetric)}
											<span
												class="inline-block size-[6px] rounded-full transition-transform duration-150"
												style={agreed ? `background: ${metricDotColor(vote)}` : `background: transparent; box-shadow: inset 0 0 0 1.5px ${metricDotColor(vote)}`}
												title={metricOutcomeText(vote)}
											></span>
										{/each}
									</div>
								{/if}
							</div>
							<svg class="h-3 w-3 flex-shrink-0 text-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
							</svg>
						</button>
					</div>
				{/each}
			</div>
			{/if}
		</section>
	{/if}

	<!-- ==================== AUDIT EVAL TAB ==================== -->
	{#if activeTab === 'audit' && hasAuditEval}
		<!-- Audit Metrics -->
		{@const auditAllMetrics = auditMetricNames.map((dim) => ({ key: dim, name: metricLabel(dim), summary: data.auditMetrics.dimensions[dim], description: data.dimensionDefs?.[dim]?.description ?? '' }))}
		<div class="mb-8 grid gap-3" style="grid-template-columns: repeat({Math.min(auditAllMetrics.length, 4)}, minmax(0, 1fr))">
			{#each auditAllMetrics as m}
				{@const pct = binaryBar(m.summary?.counts ?? { 0: 0, 1: 0 })}
				<div class="rounded-lg border border-border bg-surface px-5 py-4">
					<div class="text-[11px] font-semibold uppercase tracking-wider text-text-muted">{m.name}</div>
					{#if m.description}
					<p class="mt-0.5 text-[10px] text-text-muted/60 leading-snug line-clamp-2">{m.description}</p>
					{/if}
					<div class="mt-2 flex items-baseline gap-1.5">
						<span class="text-3xl font-bold tabular-nums {metricRateClass(m.summary?.rate ?? 0)}">{metricRateText(m.summary?.rate ?? 0)}</span>
						<span class="text-sm text-text-muted">flagged</span>
					</div>
					{#if (m.summary?.count ?? 0) > 0}
					<div class="mt-2.5 flex h-1.5 overflow-hidden rounded-full bg-border/50">
						{#if pct.clear > 0}<div class="bg-score-pass" style="width: {pct.clear}%"></div>{/if}
						{#if pct.flagged > 0}<div class="bg-score-fail" style="width: {pct.flagged}%"></div>{/if}
					</div>
					<div class="mt-1 flex justify-between text-[9px] tabular-nums text-text-muted">
						<span>{m.summary?.clear_count ?? 0} clear</span>
						<span>{m.summary?.flagged_count ?? 0} flagged</span>
						<span>{m.summary?.count ?? 0} total</span>
					</div>
					{/if}
				</div>
			{/each}
		</div>
		{#if data.auditMetrics.judgeFailures > 0}
			<p class="mb-6 text-xs text-amber-400">
				Scored {data.auditMetrics.scoredTotal} of {data.auditMetrics.total} scenarios. {data.auditMetrics.judgeFailures} judge failures were excluded from the rates.
			</p>
		{/if}

		<!-- Audit Category Accordion -->
		<section class="mb-8">
			<div class="mb-4 flex items-center gap-3">
				<h2 class="text-xs font-semibold uppercase tracking-widest text-text-muted">{auditGroupBy === 'none' ? 'All Results' : `Results by ${activeAuditAxis.label}`}</h2>
				<div class="h-px flex-1 bg-border"></div>
				<span class="text-xs text-text-muted">{data.auditScores.length} conversations{#if auditGroupBy !== 'none'} · {auditGroups.length} groups{/if}</span>
			</div>

			<!-- Controls: multi-judge filter + group-by dropdown + sort -->
			<div class="mb-3 flex items-center gap-3 flex-wrap">
				{#if hasAuditMultiJudge}
					{@const auditMjDisagreementCount = data.auditScores.filter(s => s.multi_judge && s.multi_judge.agreement < 1).length}
					<div class="flex rounded-lg bg-surface p-0.5 border border-border">
						<button
							class="px-2.5 py-1 text-[10px] font-medium rounded-md transition-colors {auditMjFilter === 'all' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}"
							onclick={() => auditMjFilter = 'all'}
						>All</button>
						<button
							class="px-2.5 py-1 text-[10px] font-medium rounded-md transition-colors {auditMjFilter === 'disagreements' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}"
							onclick={() => auditMjFilter = auditMjFilter === 'disagreements' ? 'all' : 'disagreements'}
						>Disagreements <span class="ml-1 text-zinc-600">{auditMjDisagreementCount}</span></button>
					</div>
				{/if}
				<div class="ml-auto flex items-center gap-2">
					{#if auditGroupBy === 'none'}
						<span class="text-[10px] text-text-muted">Sort by</span>
						<select
							class="rounded border border-border bg-surface px-2 py-1 text-xs text-text outline-none focus:border-interactive"
							value={auditSortMetric}
							onchange={(e) => auditSortMetric = e.currentTarget.value}
						>
							{#each auditMetricNames as m}
								<option value={m}>{metricLabel(m)}</option>
							{/each}
						</select>
					{/if}
					<span class="text-[10px] text-text-muted">Group by</span>
					<select
						class="rounded border border-border bg-surface px-2 py-1 text-xs text-text outline-none focus:border-interactive"
						value={auditGroupBy}
						onchange={(e) => setAuditGroupBy(e.currentTarget.value)}
					>
						{#each availableAxes as axis}
							<option value={axis.key} disabled={axis.disabled}>{axis.label}{axis.disabled ? ' (n/a)' : ''}</option>
						{/each}
						<option value="none">None (flat)</option>
					</select>
				</div>
			</div>

			{#if auditGroupBy !== 'none'}
			<div class="overflow-hidden rounded-lg border border-border">
				{#each auditGroups as group, gIdx (group.key)}
					<div class="{gIdx > 0 ? 'border-t border-border' : ''}">
						<!-- Group header -->
						<button
							class="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-surface {expandedAuditSubRisk === group.key ? 'bg-surface' : ''}"
							onclick={() => toggleAuditSubRisk(group.key)}
						>
							<!-- Contextual badge: show permissible/not permissible when grouping by sub_risk -->
							{#if auditGroupBy === 'sub_risk'}
								{@const groupPermissible = group.items[0]?.permissible ?? true}
								<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {groupPermissible ? 'bg-interactive/10 text-interactive' : 'bg-not-permissible/10 text-not-permissible'}">
									{groupPermissible ? 'permissible' : 'not permissible'}
								</span>
							{/if}
							<!-- Elicitation strategy badge -->
							{#if auditGroupBy === 'elicitation_strategy'}
								<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {group.key === 'base' ? 'bg-surface-2 text-text-muted' : 'bg-violet-500/10 text-violet-400'}">
									{group.key === 'base' ? 'base' : group.key}
								</span>
							{/if}
							<span class="flex-1 truncate text-sm font-medium text-text">{group.label}</span>
							<div class="flex items-center gap-2">
								{#each auditMetricNames as m}
									{#if group.avgs[m] !== undefined}
										{@const a = group.avgs[m]}
										<span class="inline-flex items-center gap-1 rounded bg-surface-2 px-1.5 py-0.5 text-[10px]" title={m.replace(/_/g, ' ')}>
											<span class="text-text-muted">{metricLabel(m)}</span>
											<span class="font-semibold tabular-nums {metricRateClass(a)}">{metricRateText(a)}</span>
										</span>
									{/if}
								{/each}
							</div>
							<span class="rounded bg-surface-2 px-2 py-0.5 text-xs tabular-nums text-text-muted">{group.total}</span>
							<svg class="h-3.5 w-3.5 flex-shrink-0 text-text-muted transition-transform duration-200 {expandedAuditSubRisk === group.key ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M9 5l7 7-7 7"/>
							</svg>
						</button>

						{#if expandedAuditSubRisk === group.key}
							<div transition:slide={{ duration: 200, easing: quintOut }} class="border-t border-border">
								{#each group.items as auditScore, sIdx}
									{@const seedInfo = data.scenarioSeedMap[auditScore.seed_id]}
									<div class="{sIdx > 0 ? 'border-t border-border/50' : ''}">
										<button
											class="flex w-full items-center gap-3 px-5 py-2.5 text-left transition-colors hover:bg-surface/50 {drawerAuditScore === auditScore ? 'bg-interactive/8 border-l-2 border-l-interactive' : ''}"
											onclick={() => openDrawer(auditScore)}
										>
											<!-- Cross-cutting context: show sub-risk + permissible when not grouping by sub_risk -->
											{#if auditGroupBy !== 'sub_risk'}
												<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {auditScore.permissible ? 'bg-interactive/10 text-interactive' : 'bg-not-permissible/10 text-not-permissible'}">{auditScore.permissible ? 'permissible' : 'not permissible'}</span>
												<span class="text-[10px] text-text-muted truncate max-w-[140px]" title={auditScore.sub_risk}>{auditScore.sub_risk}</span>
											{/if}
											<span class="flex-1 truncate text-sm text-text-secondary">{seedInfo?.title ?? auditScore.seed_id}</span>
											<!-- Variation badge after title -->
											{#if auditGroupBy !== 'elicitation_strategy' && seedInfo?.elicitation_strategy}
												<span class="flex-shrink-0 rounded px-1 py-0.5 text-[9px] font-medium bg-violet-500/10 text-violet-400">
													{seedInfo.elicitation_strategy}
												</span>
											{/if}
											<span class="text-[10px] text-text-muted tabular-nums">{auditScore.metadata.turns_count} turns</span>
											<span class="rounded bg-surface-2 px-1.5 py-0.5 text-[10px] text-text-muted">{auditScore.metadata.stop_reason}</span>
											<div class="flex items-center gap-1.5 flex-shrink-0">
												{#each auditMetricNames as m}
													{@const v = getRecordFlag(auditScore, m)}
													{#if v !== null}
														<span class="inline-flex items-center gap-1 rounded bg-surface-2 px-1.5 py-0.5 text-[10px]">
															<span class="text-text-muted">{metricLabel(m)}</span>
															<span class="font-semibold tabular-nums {metricOutcomeClass(v)}">{metricOutcomeText(v)}</span>
														</span>
													{/if}
												{/each}
												{#if judgeStatus(auditScore) === 'judge_failed'}
													<span class="inline-flex items-center rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-400">
														judge failed
													</span>
												{/if}
												{#if auditScore.multi_judge}
													<div class="flex items-center gap-0.5 ml-1" aria-label="Judge votes: {auditScore.multi_judge.votes?.[primaryAuditMetric]?.join(', ')}">
														{#each auditScore.multi_judge.votes?.[primaryAuditMetric] ?? [] as vote}
															{@const agreed = vote === getRecordFlag(auditScore, primaryAuditMetric)}
															<span
																class="inline-block size-[6px] rounded-full transition-transform duration-150"
																style={agreed ? `background: ${metricDotColor(vote)}` : `background: transparent; box-shadow: inset 0 0 0 1.5px ${metricDotColor(vote)}`}
																title={metricOutcomeText(vote)}
															></span>
														{/each}
													</div>
												{/if}
											</div>
											<svg class="h-3 w-3 flex-shrink-0 text-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
												<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
											</svg>
										</button>
									</div>
								{/each}
							</div>
						{/if}
					</div>
				{/each}
			</div>
			{:else}
			<!-- Flat list sorted by metric -->
			<div class="overflow-hidden rounded-lg border border-border">
				{#each flatAuditScores as auditScore, sIdx}
					{@const seedInfo = data.scenarioSeedMap[auditScore.seed_id]}
					<div class="{sIdx > 0 ? 'border-t border-border/50' : ''}">
						<button
							class="flex w-full items-center gap-3 px-5 py-2.5 text-left transition-colors hover:bg-surface/50 {drawerAuditScore === auditScore ? 'bg-interactive/8 border-l-2 border-l-interactive' : ''}"
							onclick={() => openDrawer(auditScore)}
						>
							<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {auditScore.permissible ? 'bg-interactive/10 text-interactive' : 'bg-not-permissible/10 text-not-permissible'}">{auditScore.permissible ? 'permissible' : 'not permissible'}</span>
							<span class="truncate text-sm text-text-secondary" style="flex: 1 1 0; min-width: 0">{seedInfo?.title ?? auditScore.seed_id}</span>
							{#if seedInfo?.elicitation_strategy}
								<span class="flex-shrink-0 rounded px-1 py-0.5 text-[9px] font-medium bg-violet-500/10 text-violet-400">
									{seedInfo.elicitation_strategy}
								</span>
							{/if}
							<span class="text-[10px] text-text-muted tabular-nums">{auditScore.metadata.turns_count} turns</span>
							<span class="rounded bg-surface-2 px-1.5 py-0.5 text-[10px] text-text-muted">{auditScore.metadata.stop_reason}</span>
							<div class="flex items-center gap-1.5 flex-shrink-0">
								{#each auditMetricNames as m}
									{@const v = getRecordFlag(auditScore, m)}
									{#if v !== null}
										<span class="inline-flex items-center gap-1 rounded bg-surface-2 px-1.5 py-0.5 text-[10px]">
											<span class="text-text-muted">{metricLabel(m)}</span>
											<span class="font-semibold tabular-nums {metricOutcomeClass(v)}">{metricOutcomeText(v)}</span>
										</span>
									{/if}
								{/each}
								{#if judgeStatus(auditScore) === 'judge_failed'}
									<span class="inline-flex items-center rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-400">
										judge failed
									</span>
								{/if}
							</div>
							<svg class="h-3 w-3 flex-shrink-0 text-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
							</svg>
						</button>
					</div>
				{/each}
			</div>
			{/if}
		</section>
	{/if}

	{#if activeTab === 'audit' && !hasAuditEval && hasAuditPreview}
		<div class="mb-6 rounded-lg border border-interactive/20 bg-interactive/5 px-5 py-4">
			<div class="text-[11px] font-semibold uppercase tracking-wider text-interactive">Rollout Preview</div>
			<p class="mt-1 text-sm text-text-secondary">
				{data.rolloutPreviewRows.length} / {data.rolloutPreviewTotal} conversations are available. Judgments will appear after rollout completes.
			</p>
		</div>

		<section class="mb-8">
			<div class="mb-4 flex items-center gap-3">
				<h2 class="text-xs font-semibold uppercase tracking-widest text-text-muted">Available Conversations</h2>
				<div class="h-px flex-1 bg-border"></div>
				<span class="text-xs text-text-muted">{data.rolloutPreviewRows.length} conversations</span>
			</div>

			<div class="overflow-hidden rounded-lg border border-border">
				{#each data.rolloutPreviewRows as preview, sIdx}
					{@const seedInfo = data.scenarioSeedMap[preview.seed_id]}
					<div class="{sIdx > 0 ? 'border-t border-border/50' : ''}">
						<button
							class="flex w-full items-center gap-3 px-5 py-2.5 text-left transition-colors hover:bg-surface/50 {drawerPreviewSeedId === preview.seed_id ? 'bg-interactive/8 border-l-2 border-l-interactive' : ''}"
							onclick={() => openPreviewDrawer(preview)}
						>
							<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {preview.permissible ? 'bg-interactive/10 text-interactive' : 'bg-not-permissible/10 text-not-permissible'}">
								{preview.permissible ? 'permissible' : 'not permissible'}
							</span>
							<div class="min-w-0 flex-1">
								<div class="truncate text-sm text-text-secondary">{seedInfo?.title ?? preview.seed_id}</div>
								<div class="mt-0.5 truncate text-[10px] text-text-muted" title={preview.sub_risk}>{preview.sub_risk}</div>
							</div>
							{#if seedInfo?.elicitation_strategy}
								<span class="flex-shrink-0 rounded px-1 py-0.5 text-[9px] font-medium bg-violet-500/10 text-violet-400">
									{seedInfo.elicitation_strategy}
								</span>
							{/if}
							<span class="text-[10px] text-text-muted tabular-nums">{preview.turns_count} turns</span>
							<span class="rounded bg-surface-2 px-1.5 py-0.5 text-[10px] text-text-muted">{preview.stop_reason}</span>
							<span class="inline-flex items-center rounded bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium text-text-muted">
								unjudged
							</span>
							<svg class="h-3 w-3 flex-shrink-0 text-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
								<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
							</svg>
						</button>
					</div>
				{/each}
			</div>
		</section>
	{/if}
{/if}

<!-- Unified detail modal -->
{#if drawerItem}
	<ResultDrawer
		item={drawerItem}
		metricNames={drawerMetricNames}
		primaryMetric={drawerPrimaryMetric}
		requiredBaseMetrics={requiredBaseMetrics}
		navIdx={drawerNavIdx}
		navTotal={drawerNavTotal}
		onClose={closeActiveDrawer}
		onPrev={() => navigateActiveDrawer(-1)}
		onNext={() => navigateActiveDrawer(1)}
	/>
{/if}
