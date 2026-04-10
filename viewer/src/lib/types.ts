// Suite-level types (from suite.json, policy.json, seeds.jsonl)

export interface Suite {
	created_at: string;
}

export interface SubRisk {
	name: string;
	definition: string;
	examples: string[];
	permissible: boolean;
}

export interface Policy {
	risk: {
		name: string;
		definition: string;
	};
	definition_of_terms?: Array<{
		term: string;
		definition: string;
		examples: string[];
	}>;
	sub_risks: SubRisk[];
}

export interface SeedPayload {
	title: string;
	description: string;
	system_prompt?: string | null;
	tools?: SeedTool[];
}

export interface SeedToolParameter {
	name: string;
	type?: string;
	description?: string;
}

export interface SeedTool {
	name: string;
	description: string;
	parameters?: SeedToolParameter[];
}

export interface PromptSeed {
	seed_id: string;
	permissible: boolean;
	risk: string;
	sub_risk: string;
	definition: string;
	seed: SeedPayload;
}

export interface ScenarioSeed {
	seed_id: string;
	parent_seed_id?: string | null;
	elicitation_strategy?: string | null;
	risk: string;
	sub_risk: string;
	definition: string;
	permissible: boolean;
	seed: SeedPayload;
}

export interface ViewerSeedItem {
	id: string;
	kind: 'prompt' | 'scenario';
	title: string;
	description: string;
	sub_risk: string;
	definition: string;
	permissible: boolean;
	system_prompt?: string | null;
	tools?: SeedPayload['tools'];
	elicitation_strategy?: string | null;
}

export interface ViewerSeedGroup {
	name: string;
	permissible: boolean;
	definition: string;
	items: ViewerSeedItem[];
}

// Run-level types (from manifest.json, scores.jsonl + transcripts.jsonl)

export type StageStatus = 'running' | 'completed' | 'failed';

export interface Manifest {
	status?: 'running' | 'completed' | 'failed';
	started_at?: string;
	ended_at?: string;
	stages?: Record<string, StageStatus>;
}

export interface NodeJudgment {
	node_index: number;
	node_name?: string;
	relevant: boolean;
	violated: boolean | null;
	confidence: 'high' | 'medium' | 'low';
	evidence_turns: number[];
	reasoning: string;
}

export interface LegacyPromptCitation {
	quote: string;
	relevance?: string;
}

export interface Verdict {
	dimensions: Record<string, boolean>;
	justification: string;
	dimension_justifications?: Record<string, string>;
	node_judgments: NodeJudgment[];
	citations?: AuditCitation[] | LegacyPromptCitation[];
	citation_warnings?: string[];
	judgment_warnings?: string[];
	[key: string]: unknown;
}

export interface JudgmentErrorVerdict {
	error: string;
	[key: string]: unknown;
}

export type JudgeStatus = 'ok' | 'judge_failed';

export interface MultiJudge {
	n: number;
	n_failed: number;
	votes: Record<string, boolean[]>;
	means: Record<string, number>;
	agreement: number;
	justifications: string[];
	representative_index?: number;
	verdicts?: Verdict[];
}

export interface ToolCallMessage {
	id?: string;
	function: string;
	arguments: Record<string, unknown>;
}

export interface InteractionMessage {
	id?: string;
	role: 'system' | 'user' | 'assistant' | 'tool';
	content: string;
	type?: string;
	judgeTurn?: number | null;
	tool_calls?: ToolCallMessage[];
	tool_call_id?: string;
	function?: string;
	arguments?: Record<string, unknown>;
	raw?: Record<string, unknown>;
}

export interface LlmCallTrace {
	call_id: string;
	source: string;
	api_mode: string;
	request: unknown;
	response: unknown;
	derived?: Record<string, unknown>;
	message_ids: string[];
}

export interface JudgedSample {
	prompt: string;
	response: string;
	risk?: string | null;
	sub_risk: string;
	permissible?: boolean | null;
	run_id?: string;
	judge_model?: string;
	target?: string;
	epoch?: number | null;
	seed_metadata?: Record<string, unknown> | null;
	verdict?: Verdict | JudgmentErrorVerdict | null;
	judge_status?: JudgeStatus | null;
	judge_error?: string | null;
	messages?: InteractionMessage[];
	llm_calls?: LlmCallTrace[];
	target_runtime_mode?: string | null;
	multi_judge?: MultiJudge;
}

export interface ViewerResultContext {
	description?: string | null;
	tools?: SeedPayload['tools'];
	turns_count?: number | null;
	stop_reason?: string | null;
	elicitation_strategy?: string | null;
	parent_title?: string | null;
}

export interface ViewerResultItem {
	id: string;
	kind: 'prompt' | 'scenario';
	row_title: string;
	header_title: string;
	sub_risk: string;
	permissible: boolean;
	verdict?: Verdict | AuditVerdict | JudgmentErrorVerdict | null;
	judge_status?: JudgeStatus | null;
	judge_error?: string | null;
	multi_judge?: MultiJudge;
	messages: InteractionMessage[];
	llm_calls: LlmCallTrace[];
	target_runtime_mode?: string | null;
	context: ViewerResultContext;
}

// Aggregated types for views

export type SuiteStatus = 'policy_only' | 'seeds_ready' | 'has_results';

export interface SuiteListItem {
	suite_id: string;
	risk_name: string;
	sub_risk_count: number;
	seed_count: number;
	scenario_seed_count: number;
	run_count: number;
	runs: string[];
	status: SuiteStatus;
	created_at: string;
	has_systematization: boolean;
}

export interface DimensionDef {
	description: string;
	rubric: string;
	kind: 'event';
	polarity?: 'negative' | 'positive';
	required_base?: boolean;
}

export interface BinaryCounts {
	0: number;
	1: number;
}

export interface DimensionMetrics {
	rate: number;
	count: number;
	flagged_count: number;
	clear_count: number;
	counts: BinaryCounts;
}

export interface RunMetrics {
	total: number;
	scored_total: number;
	judge_failures: number;
	judge_failure_rate: number;
	counts: BinaryCounts;
	policy_violation_rate: number;
	overrefusal_rate: number;
	permissible_overrefusal_rate: number;
	not_permissible_policy_violation_rate: number;
	target: string;
	judge_model: string;
	dimensions: Record<string, DimensionMetrics>;
}

export interface RunListItem {
	run_id: string;
	has_judged: boolean;
	has_scenario_scores: boolean;
	manifest: Manifest | null;
	metrics: RunMetrics | null;
}

export interface AuditRunListItem {
	run_id: string;
	has_scores: boolean;
	manifest: Manifest | null;
	metrics: AuditRunMetrics | null;
}

export interface AuditRunMetrics {
	total: number;
	scored_total: number;
	judge_failures: number;
	judge_failure_rate: number;
	counts: BinaryCounts;
	policy_violation_rate: number;
	overrefusal_rate: number;
	dimensions: Record<string, DimensionMetrics>;
	target: string;
	auditor_model: string;
	judge_model: string;
}

export interface ScoreDistribution {
	sub_risk: string;
	permissible: boolean;
	counts: BinaryCounts;
	total: number;
}

// --- Audit eval types ---

export type AuditCitationResolutionStatus = 'resolved' | 'unresolved' | 'ambiguous';

export type AuditCitationResolutionMethod =
	| 'raw_exact'
	| 'normalized_exact'
	| 'neighbor_raw_exact'
	| 'neighbor_normalized_exact'
	| 'transcript_raw_exact'
	| 'transcript_normalized_exact'
	| 'invalid_position'
	| 'quote_not_found'
	| 'ambiguous_quote_match'
	| 'conservative_fuzzy'
	| 'missing_message_id'
	| 'missing_message_text'
	| 'missing_resolution';

export interface AuditCitationAnchor {
	exact: string;
	prefix?: string | null;
	suffix?: string | null;
	hint?: number | null;
}

export interface AuditCitationResolution {
	status: AuditCitationResolutionStatus;
	method: AuditCitationResolutionMethod;
	detail?: string | null;
}

export type AuditCitationSourceKind = 'message' | 'tool_arg' | 'tool_result';

export interface AuditCitationPart {
	claimed_message_index?: string | null;
	matched_message_index?: string | null;
	message_id?: string | null;
	source_kind?: AuditCitationSourceKind | null;
	tool_call_id?: string | null;
	tool_arg?: string | null;
	quoted_text: string;
	position: [number, number] | null;
	anchor?: AuditCitationAnchor | null;
	resolution?: AuditCitationResolution | null;
}

export interface AuditCitation {
	index?: number | null;
	description: string;
	parts: AuditCitationPart[];
}

export interface AuditVerdict {
	dimensions: Record<string, boolean>;
	justification: string;
	dimension_justifications?: Record<string, string>;
	node_judgments: NodeJudgment[];
	citations?: AuditCitation[] | LegacyPromptCitation[];
	citation_warnings?: string[];
	judgment_warnings?: string[];
	[key: string]: unknown;
}

export interface AuditTranscriptMessage {
	id?: string;
	role: string;
	content: string;
	type?: string;
	judgeTurn?: number | null;
	tool_calls?: ToolCallMessage[];
	tool_call_id?: string;
	function?: string;
	arguments?: Record<string, unknown>;
	raw?: Record<string, unknown>;
}

export interface AuditTranscriptEvent {
	view: string[];
	actor: string;
	raw?: Record<string, unknown>;
	edit: {
		type: string;
		message: AuditTranscriptMessage;
		tool_name?: string;
		tool_args?: Record<string, unknown>;
		tool_call_id?: string;
		tool_result?: string;
	};
}

export interface AuditTranscript {
	seed_id: string;
	risk: string;
	sub_risk: string;
	permissible: boolean;
	events: AuditTranscriptEvent[];
	llm_calls?: LlmCallTrace[];
	stop_reason: string;
	target: string;
	auditor_model: string;
}

export interface AuditScore {
	seed_id: string;
	risk: string;
	sub_risk: string;
	permissible: boolean;
	judge_model: string;
	target?: string;
	auditor_model?: string;
	verdict?: AuditVerdict | JudgmentErrorVerdict | null;
	judge_status?: JudgeStatus | null;
	judge_error?: string | null;
	target_runtime_mode?: string | null;
	metadata: {
		turns_count: number;
		stop_reason: string;
	};
	multi_judge?: MultiJudge;
}

// --- Grouping types ---

export interface ScenarioSeedInfo {
	title: string;
	description: string;
	tools?: SeedTool[];
	parent_seed_id?: string | null;
	elicitation_strategy?: string | null;
	target_runtime_mode?: string | null;
}

export interface GroupContext {
	scenarioSeedMap: Record<string, ScenarioSeedInfo>;
}

export interface GroupAxis<T> {
	key: string;
	label: string;
	accessor: (item: T, context?: GroupContext) => string;
	sortGroups?: (a: GroupEntry<T>, b: GroupEntry<T>) => number;
}

export interface GroupEntry<T> {
	key: string;
	label: string;
	items: T[];
	avgs: Record<string, number>;
	total: number;
}
