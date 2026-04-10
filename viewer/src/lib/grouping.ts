/**
 * Generic grouping system for audit scores.
 * New grouping axes can be added by pushing to AUDIT_GROUP_AXES.
 */

import { getVerdictFlag, scoreSortValue } from './judgment.js';
import type { AuditScore, GroupAxis, GroupContext, GroupEntry } from './types.js';

// ---------------------------------------------------------------------------
// Axis registry
// ---------------------------------------------------------------------------

export const AUDIT_GROUP_AXES: GroupAxis<AuditScore>[] = [
	{
		key: 'sub_risk',
		label: 'Sub-risk',
		accessor: (s) => s.sub_risk,
		sortGroups: (a, b) => {
			const aPermissible = a.items[0]?.permissible ?? true;
			const bPermissible = b.items[0]?.permissible ?? true;
			if (aPermissible !== bPermissible) return aPermissible ? 1 : -1;
			return a.key.localeCompare(b.key);
		},
	},
	{
		key: 'elicitation_strategy',
		label: 'Elicitation strategy',
		accessor: (s, ctx) => {
			const seedInfo = ctx?.scenarioSeedMap[s.seed_id];
			return seedInfo?.elicitation_strategy ?? 'base';
		},
	},
	{
		key: 'permissible',
		label: 'Permissibility',
		accessor: (s) => (s.permissible ? 'Permissible' : 'Not permissible'),
		sortGroups: (a, b) => {
			// "Not permissible" first
			if (a.key !== b.key) return a.key === 'Not permissible' ? -1 : 1;
			return 0;
		},
	},
	{
		key: 'seed_family',
		label: 'Seed family',
		accessor: (s, ctx) => {
			const seedInfo = ctx?.scenarioSeedMap[s.seed_id];
			return seedInfo?.parent_seed_id ?? s.seed_id;
		},
	},
];

// ---------------------------------------------------------------------------
// Generic group-by function
// ---------------------------------------------------------------------------

export function groupByAxis<T extends { verdict?: Record<string, unknown> | null }>(
	items: T[],
	axis: GroupAxis<T>,
	metricNames: string[],
	context?: GroupContext,
): GroupEntry<T>[] {
	const map = new Map<string, { items: T[]; sums: Record<string, number>; counts: Record<string, number> }>();

	for (const item of items) {
		const key = axis.accessor(item, context) ?? '(ungrouped)';
		if (!map.has(key)) {
			map.set(key, { items: [], sums: {}, counts: {} });
		}
		const bucket = map.get(key)!;
		bucket.items.push(item);

		for (const m of metricNames) {
			const flag = getVerdictFlag(item.verdict, m);
			if (flag !== null) {
				bucket.sums[m] = (bucket.sums[m] ?? 0) + Number(flag);
				bucket.counts[m] = (bucket.counts[m] ?? 0) + 1;
			}
		}
	}

	const entries: GroupEntry<T>[] = [...map.entries()].map(([key, g]) => {
		const avgs: Record<string, number> = {};
		for (const m of metricNames) {
			if (g.counts[m]) avgs[m] = g.sums[m] / g.counts[m];
		}
		return {
			key,
			label: key,
			items: [...g.items].sort(
				(a, b) => {
					return scoreSortValue(a as { verdict?: Record<string, unknown> | null }, 'policy_violation')
						- scoreSortValue(b as { verdict?: Record<string, unknown> | null }, 'policy_violation');
				}
			),
			avgs,
			total: g.items.length,
		};
	});

	if (axis.sortGroups) {
		entries.sort(axis.sortGroups);
	} else {
		entries.sort((a, b) => a.key.localeCompare(b.key));
	}

	return entries;
}
