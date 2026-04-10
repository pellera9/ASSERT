import type {
	AuditRunListItem,
	PromptSeed,
	RunListItem,
	ScenarioSeed,
	SubRisk,
	ViewerSeedGroup,
	ViewerSeedItem
} from '$lib/types.js';

export interface CombinedRunEntry {
	run_id: string;
	compare_run_id: string | null;
	query_run_id: string | null;
	audit_run_id: string | null;
	query: RunListItem | null;
	audit: AuditRunListItem | null;
}

function matchesPermissibleFilter(value: boolean, filter: string): boolean {
	return filter === 'all' || String(value) === filter;
}

export function normalizePromptSeeds(items: PromptSeed[]): ViewerSeedItem[] {
	return items.map((seed) => ({
		id: seed.seed_id,
		kind: 'prompt',
		title: seed.seed.title || seed.sub_risk,
		description: seed.seed.description,
		sub_risk: seed.sub_risk,
		definition: seed.definition,
		permissible: seed.permissible,
		system_prompt: seed.seed.system_prompt ?? null,
		tools: seed.seed.tools
	}));
}

export function normalizeScenarioSeeds(items: ScenarioSeed[]): ViewerSeedItem[] {
	return items.map((seed) => ({
		id: seed.seed_id,
		kind: 'scenario',
		title: seed.seed.title,
		description: seed.seed.description,
		sub_risk: seed.sub_risk,
		definition: seed.definition,
		permissible: seed.permissible,
		system_prompt: seed.seed.system_prompt ?? null,
		tools: seed.seed.tools,
		elicitation_strategy: seed.elicitation_strategy ?? null
	}));
}

export function filterViewerSeeds(
	items: ViewerSeedItem[],
	query: string,
	permissibleFilter: string
): ViewerSeedItem[] {
	let filteredSeeds = items;
	if (query) {
		const normalizedQuery = query.toLowerCase();
		filteredSeeds = filteredSeeds.filter(
			(seed) =>
				seed.title.toLowerCase().includes(normalizedQuery) ||
				seed.description.toLowerCase().includes(normalizedQuery) ||
				seed.sub_risk.toLowerCase().includes(normalizedQuery)
		);
	}
	return filteredSeeds.filter((seed) => matchesPermissibleFilter(seed.permissible, permissibleFilter));
}

export function groupViewerSeedsByPolicy(
	items: ViewerSeedItem[],
	subRisks: SubRisk[]
): ViewerSeedGroup[] {
	const groupedSeeds = new Map<string, ViewerSeedItem[]>();
	for (const seed of items) {
		if (!groupedSeeds.has(seed.sub_risk)) groupedSeeds.set(seed.sub_risk, []);
		groupedSeeds.get(seed.sub_risk)!.push(seed);
	}

	const orderedGroups: ViewerSeedGroup[] = [];
	for (const subRisk of subRisks) {
		const matchingSeeds = groupedSeeds.get(subRisk.name);
		if (!matchingSeeds) continue;
		orderedGroups.push({
			name: subRisk.name,
			permissible: subRisk.permissible,
			definition: subRisk.definition,
			items: matchingSeeds
		});
		groupedSeeds.delete(subRisk.name);
	}

	for (const [name, remainingSeeds] of groupedSeeds) {
		orderedGroups.push({
			name,
			permissible: remainingSeeds[0]?.permissible ?? true,
			definition: remainingSeeds[0]?.definition ?? '',
			items: remainingSeeds
		});
	}

	return orderedGroups;
}

export function mergeRunLists(
	runs: RunListItem[],
	auditRuns: AuditRunListItem[]
): CombinedRunEntry[] {
	const combinedRuns = new Map<string, CombinedRunEntry>();

	for (const run of runs) {
		combinedRuns.set(run.run_id, {
			run_id: run.run_id,
			compare_run_id: run.run_id,
			query_run_id: run.run_id,
			audit_run_id: null,
			query: run,
			audit: null
		});
	}

	for (const auditRun of auditRuns) {
		const existing = combinedRuns.get(auditRun.run_id);
		if (existing) {
			existing.audit = auditRun;
			existing.audit_run_id = auditRun.run_id;
			continue;
		}

		combinedRuns.set(auditRun.run_id, {
			run_id: auditRun.run_id,
			compare_run_id: null,
			query_run_id: null,
			audit_run_id: auditRun.run_id,
			query: null,
			audit: auditRun
		});
	}

	return [...combinedRuns.values()];
}
