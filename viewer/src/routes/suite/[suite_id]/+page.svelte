<script lang="ts">
import type { Policy } from '$lib/types.js';
import { invalidateAll, replaceState } from '$app/navigation';
import { page } from '$app/state';
import { slide } from 'svelte/transition';
import { quintOut } from 'svelte/easing';
import { renderMarkdown } from '$lib/markdown';
import SeedGroupList from '$lib/SeedGroupList.svelte';
import SystematizationModal from '$lib/SystematizationModal.svelte';
import {
	filterViewerSeeds,
	mergeRunLists,
	groupViewerSeedsByPolicy,
	normalizePromptSeeds,
	normalizeScenarioSeeds
} from '$lib/suite-view.js';

let { data } = $props();

// Tab state — initialize from URL ?section= param
type Tab = 'policy' | 'seeds' | 'results';
const VALID_TABS = new Set<string>(['policy', 'seeds', 'results']);
let initialTab = page.url.searchParams.get('section');
let activeTab = $state<Tab | null>(VALID_TABS.has(initialTab ?? '') ? initialTab as Tab : null);

// Sync activeTab to URL
$effect(() => {
	const url = new URL(page.url);
	if (activeTab) {
		url.searchParams.set('section', activeTab);
	} else {
		url.searchParams.delete('section');
	}
	replaceState(url, {});
});

// Description expand
let descExpanded = $state(false);

// Metadata panel
let metaOpen = $state(false);

// Systematization expand
let systematizationModalOpen = $state(false);

// Comparison selection
let selectedRuns = $state<Set<string>>(new Set());

function toggleRunSelection(runId: string | null) {
	if (!runId) return;
	const next = new Set(selectedRuns);
	if (next.has(runId)) next.delete(runId);
	else next.add(runId);
	selectedRuns = next;
}

let canCompare = $derived(selectedRuns.size >= 2 && selectedRuns.size <= 4);

// Policy editing
interface EditableSubRisk {
	name: string;
	definition: string;
	examples: string[];
	permissible: boolean;
}
type EditablePolicy = Omit<Policy, 'sub_risks'> & { sub_risks: EditableSubRisk[] };

function cloneEditablePolicy(): EditablePolicy | null {
	if (!data.policy) return null;
	return structuredClone(data.policy) as EditablePolicy;
}

let editModalOpen = $state(false);
let editingIndex = $state<number | null>(null); // null = adding new
let editForm = $state<EditableSubRisk>({ name: '', definition: '', examples: [], permissible: false });
let editExamplesText = $state('');
let editSaving = $state(false);
let editError = $state<string | null>(null);
let deleteConfirmIndex = $state<number | null>(null);
let seedsWarningPending = $state(false);
let pendingPolicy = $state<Record<string, unknown> | null>(null);

function openEditModal(idx: number) {
	const sr = sortedSubRisks[idx];
	editingIndex = idx;
	editForm = { name: sr.name, definition: sr.definition, examples: [...sr.examples], permissible: sr.permissible };
	editExamplesText = sr.examples.join('\n');
	editError = null;
	editModalOpen = true;
}

function openAddModal() {
	editingIndex = null;
	editForm = { name: '', definition: '', examples: [], permissible: false };
	editExamplesText = '';
	editError = null;
	editModalOpen = true;
}

function closeEditModal() {
	editModalOpen = false;
	editingIndex = null;
	editError = null;
}

async function savePolicy(policy: Record<string, unknown>) {
	editSaving = true;
	editError = null;
	try {
		const res = await fetch('/api/policy', {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ suite_id: data.suite_id, policy })
		});
		const body = await res.json();
		if (!res.ok) { editError = body.error ?? 'Failed to save'; editSaving = false; return false; }
		editSaving = false;
		return true;
	} catch (e: unknown) {
		editError = e instanceof Error ? e.message : 'Failed to save';
		editSaving = false;
		return false;
	}
}

async function handleSaveSubRisk() {
	const examples = editExamplesText.split('\n').map(e => e.trim()).filter(Boolean);
	if (!editForm.name.trim()) { editError = 'Name is required'; return; }
	if (!editForm.definition.trim()) { editError = 'Definition is required'; return; }

	const currentTax = cloneEditablePolicy();
	if (!currentTax) return;
	const subRisks = currentTax.sub_risks;
	const entry = { name: editForm.name.trim(), definition: editForm.definition.trim(), examples, permissible: editForm.permissible };

	if (editingIndex !== null) {
		const origName = sortedSubRisks[editingIndex].name;
		const realIdx = subRisks.findIndex(sr => sr.name === origName);
		if (realIdx >= 0) subRisks[realIdx] = entry;
	} else {
		if (subRisks.some(sr => sr.name.toLowerCase() === entry.name.toLowerCase())) {
			editError = 'A category with this name already exists'; return;
		}
		subRisks.push(entry);
	}

	const hasSeeds = data.promptSeeds.length > 0 || data.scenarioSeeds.length > 0;
	if (hasSeeds) {
		pendingPolicy = currentTax;
		seedsWarningPending = true;
		closeEditModal();
		return;
	}

	const ok = await savePolicy(currentTax);
	if (ok) { closeEditModal(); await invalidateAll(); }
}

async function handleDeleteSubRisk(idx: number) {
	const currentTax = cloneEditablePolicy();
	if (!currentTax) return;
	const subRisks = currentTax.sub_risks;
	const origName = sortedSubRisks[idx].name;
	const realIdx = subRisks.findIndex(sr => sr.name === origName);
	if (realIdx >= 0) subRisks.splice(realIdx, 1);

	const hasSeeds = data.promptSeeds.length > 0 || data.scenarioSeeds.length > 0;
	if (hasSeeds) {
		pendingPolicy = currentTax;
		seedsWarningPending = true;
		deleteConfirmIndex = null;
		return;
	}

	const ok = await savePolicy(currentTax);
	if (ok) { deleteConfirmIndex = null; await invalidateAll(); }
}

async function confirmSaveWithSeeds() {
	if (!pendingPolicy) return;
	const ok = await savePolicy(pendingPolicy);
	if (ok) { seedsWarningPending = false; pendingPolicy = null; await invalidateAll(); }
}

// Seeds sub-tab
let seedsSubTab = $state<'query' | 'scenarios'>('query');

let expandedSubRisk = $state<string | null>(null);
let expandedPromptSeedSubRisk = $state<string | null>(null);
let promptSeedFilter = $state('');
let promptSeedPermissibleFilter = $state<string>('all');

let expandedAuditSubRisk = $state<string | null>(null);
let scenarioSeedFilter = $state('');
let scenarioSeedPermissibleFilter = $state<string>('all');

function toggle(name: string) {
	expandedSubRisk = expandedSubRisk === name ? null : name;
}

function togglePromptSeedSubRisk(name: string) {
	expandedPromptSeedSubRisk = expandedPromptSeedSubRisk === name ? null : name;
}

function toggleAuditSubRisk(name: string) {
	expandedAuditSubRisk = expandedAuditSubRisk === name ? null : name;
}

let promptSeedItems = $derived(normalizePromptSeeds(data.promptSeeds));
let scenarioSeedItems = $derived(normalizeScenarioSeeds(data.scenarioSeeds));

let filteredPromptSeeds = $derived.by(() => {
	return filterViewerSeeds(promptSeedItems, promptSeedFilter, promptSeedPermissibleFilter);
});

let sortedSubRisks = $derived(data.policy?.sub_risks ?? []);

let promptSeedsBySubRisk = $derived.by(() => {
	return groupViewerSeedsByPolicy(filteredPromptSeeds, sortedSubRisks);
});

let filteredScenarioSeeds = $derived.by(() => {
	return filterViewerSeeds(scenarioSeedItems, scenarioSeedFilter, scenarioSeedPermissibleFilter);
});

let scenarioSeedsBySubRisk = $derived.by(() => {
	return groupViewerSeedsByPolicy(filteredScenarioSeeds, sortedSubRisks);
});

// Truncated description
let riskDef = $derived(data.policy?.risk?.definition ?? '');
let needsTruncation = $derived(riskDef.length > 120);
let displayDef = $derived(needsTruncation && !descExpanded ? riskDef.slice(0, 120) + '…' : riskDef);

function summaryItemCountFor(systematization: Record<string, unknown> | null): number {
	if (!systematization) return 0;
	return Array.isArray(systematization.summary_items) ? systematization.summary_items.length : 0;
}

function systematizationModeFor(systematization: Record<string, unknown> | null): string | null {
	if (!systematization) return null;
	const meta = systematization.meta;
	if (!meta || typeof meta !== 'object') return null;
	return typeof (meta as Record<string, unknown>).mode === 'string'
		? String((meta as Record<string, unknown>).mode)
		: null;
}

// Systematization
let hasSystematization = $derived(!!data.systematization);
let summaryItemCount = $derived(
	summaryItemCountFor((data.systematization as Record<string, unknown> | null) ?? null)
);
let systematizationMode = $derived(
	systematizationModeFor((data.systematization as Record<string, unknown> | null) ?? null)
);

const TAB_DESCRIPTIONS: Record<string, string> = {
	policy: 'Risk policy broken into categories with definitions and examples',
	seeds: 'Test seeds and multi-turn scenarios used to evaluate the model',
	results: 'Results from measurement runs',
};

// Merge query/audit results that share the same run id.
let allRuns = $derived.by(() => {
	return mergeRunLists(data.runs, data.auditRuns);
});

const TAB_ICONS: Record<string, string> = {
	policy: '<path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>',
	seeds: '<path d="M4 6h16M4 10h16M4 14h16M4 18h16"/>',
	results: '<path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>',
};

const tabs: { key: Tab; label: string; count: number }[] = $derived([
	{ key: 'policy', label: 'Policy', count: sortedSubRisks.length },
	{ key: 'seeds', label: 'Seeds', count: data.promptSeeds.length + data.scenarioSeeds.length },
	{ key: 'results', label: 'Results', count: allRuns.length },
]);


</script>

<!-- Header -->
<div class="mb-6">
<a href="/" class="group inline-flex items-center gap-1.5 text-xs text-text-muted transition-colors hover:text-interactive">
<svg class="h-3 w-3 transition-transform group-hover:-translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M15 19l-7-7 7-7"/></svg>
All measurement suites
</a>
<div class="text-center">
<h1 class="mt-2 text-xl font-semibold tracking-tight">{data.policy?.risk?.name ?? data.suite_id}</h1>
<span class="mt-1.5 inline-block rounded bg-surface-2 px-2 py-0.5 font-mono text-[11px] text-text-muted">{data.suite_id}</span>
{#if data.policy?.risk}
<p class="mx-auto mt-2 max-w-2xl text-sm text-text-secondary leading-relaxed">
{displayDef}
{#if needsTruncation}
<button class="ml-1 text-interactive hover:text-interactive-hover text-xs" onclick={() => descExpanded = !descExpanded}>
{descExpanded ? 'show less' : 'show more'}
</button>
{/if}
</p>
{/if}
<div class="mt-3 flex items-center justify-center gap-2">
<span class="inline-flex items-center gap-1.5 rounded-full bg-surface px-2.5 py-1 text-xs text-text-muted">
<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
{data.suite?.created_at ? new Date(data.suite.created_at).toLocaleDateString() : '—'}
</span>
{#if hasSystematization}
<button
class="inline-flex items-center gap-1 rounded-full bg-surface px-2.5 py-1 text-xs text-text-muted transition-colors hover:text-text-secondary"
onclick={() => metaOpen = !metaOpen}
title="Suite metadata"
>
<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
{metaOpen ? 'hide details' : 'details'}
</button>
{/if}
</div>
{#if metaOpen && (data.suite || hasSystematization)}
<div class="mx-auto mt-3 max-w-2xl rounded-lg border border-border bg-surface p-4 text-left" transition:slide={{ duration: 200, easing: quintOut }}>
{#if hasSystematization}
<div class="mt-3 border-t border-border pt-3">
<h4 class="mb-1.5 text-xs font-semibold uppercase tracking-wider text-text-muted">Systematization Artifacts</h4>
<div class="flex flex-wrap gap-x-4 gap-y-1">
<span class="text-xs text-text-muted"><span class="text-text-secondary">systematization:</span> present</span>
{#if systematizationMode}
<span class="text-xs text-text-muted"><span class="text-text-secondary">mode:</span> {systematizationMode}</span>
{/if}
{#if summaryItemCount > 0}
<span class="text-xs text-text-muted"><span class="text-text-secondary">pattern summaries:</span> {summaryItemCount}</span>
{/if}
<span class="text-xs text-text-muted"><span class="text-text-secondary">policy categories:</span> {sortedSubRisks.length}</span>
</div>
</div>
{/if}
</div>
{/if}
</div>
</div>

<!-- Section cards -->
<div class="mb-6 grid gap-4 sm:grid-cols-3">
{#each tabs as tab}
<button
class="group rounded-lg border p-5 text-left transition-all {activeTab === tab.key ? 'border-interactive bg-surface shadow-sm' : 'border-border bg-surface hover:border-interactive/50 hover:shadow-sm'}"
onclick={() => activeTab = activeTab === tab.key ? null : tab.key}
>
<div class="mb-3 flex items-center justify-between">
<div class="flex items-center gap-2.5">
<div class="flex h-8 w-8 items-center justify-center rounded-lg {activeTab === tab.key ? 'bg-interactive/10' : 'bg-surface-2'}">
<svg class="h-4 w-4 {activeTab === tab.key ? 'text-interactive' : 'text-text-muted'}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">{@html TAB_ICONS[tab.key]}</svg>
</div>
<h2 class="text-sm font-semibold {activeTab === tab.key ? 'text-interactive' : 'text-text'}">{tab.label}</h2>
</div>
<span class="font-mono text-lg font-semibold text-text-secondary">{tab.count}</span>
</div>
<p class="text-xs text-text-muted leading-relaxed">{TAB_DESCRIPTIONS[tab.key]}</p>
</button>
{/each}
</div>

{#if activeTab !== null}
<!-- Tab: Policy -->
{#if activeTab === 'policy'}
{#if hasSystematization}
<!-- Systematization banner -->
<div class="mb-4 rounded-lg border border-border bg-surface p-4">
<div class="flex items-center gap-3 text-xs text-text-muted">
<span class="text-text-secondary font-medium">{sortedSubRisks.length > 0 ? 'Policy generated via' : 'Systematization available'}</span>
<div class="flex items-center gap-1.5">
<button
class="inline-flex items-center gap-1 rounded-full bg-surface-2 px-2.5 py-1 font-medium text-text-secondary hover:text-text border border-transparent hover:border-interactive/30 transition-colors"
onclick={() => systematizationModalOpen = true}
>
<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
Systematization
</button>
{#if sortedSubRisks.length > 0}
<svg class="h-3 w-3 text-text-muted/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M9 5l7 7-7 7"/></svg>
<span class="inline-flex items-center gap-1 rounded-full bg-interactive/10 border border-interactive/20 px-2.5 py-1 font-medium text-interactive">
<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
Policy
</span>
{/if}
</div>
</div>
</div>
{/if}
{#if sortedSubRisks.length === 0}
<div class="rounded-lg border border-border bg-surface px-6 py-10 text-center">
<svg class="mx-auto mb-3 h-8 w-8 text-text-muted opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
<path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
</svg>
<p class="text-sm text-text-secondary">No policy generated yet.</p>
<p class="mt-1 text-xs text-text-muted">Run the pipeline to generate a risk policy.</p>
</div>
{:else}
<div class="overflow-hidden rounded-lg border border-border">
{#each sortedSubRisks as sr, idx (sr.name)}
<div class="{idx > 0 ? 'border-t border-border' : ''}">
<div class="flex w-full items-center gap-3 px-4 py-2.5 text-sm">
<button
class="flex flex-1 items-center gap-3 text-left transition-colors hover:text-interactive"
onclick={() => toggle(sr.name)}
>
<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {sr.permissible ? 'bg-interactive/10 text-interactive' : 'bg-not-permissible/10 text-not-permissible'}">
{sr.permissible ? 'permissible' : 'not permissible'}
</span>
<span class="flex-1 truncate font-medium">{sr.name}</span>
<svg class="h-3.5 w-3.5 text-text-muted transition-transform duration-200 {expandedSubRisk === sr.name ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
<path d="M9 5l7 7-7 7"/>
</svg>
</button>
<button onclick={(e) => { e.stopPropagation(); openEditModal(idx); }} class="flex-shrink-0 rounded p-1 text-text-muted hover:text-interactive hover:bg-interactive/10 transition-colors" title="Edit">
<svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>
</button>
<button onclick={(e) => { e.stopPropagation(); deleteConfirmIndex = idx; }} class="flex-shrink-0 rounded p-1 text-text-muted hover:text-not-permissible hover:bg-not-permissible/10 transition-colors" title="Delete">
<svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
</button>
</div>
{#if expandedSubRisk === sr.name}
<div transition:slide={{ duration: 200, easing: quintOut }} class="border-t border-border bg-surface px-5 py-5">
<!-- Definition -->
<div class="mb-4">
<h4 class="mb-1.5 text-xs font-semibold uppercase tracking-wider text-text-muted">Definition</h4>
<div class="prose text-sm text-text-secondary leading-relaxed">{@html renderMarkdown(sr.definition)}</div>
</div>

<!-- Examples -->
{#if sr.examples?.length}
<div>
<h4 class="mb-1.5 text-xs font-semibold uppercase tracking-wider text-text-muted">Examples</h4>
<div class="space-y-1.5">
{#each sr.examples as ex}
<div class="border-l-2 border-border pl-3 text-sm text-text-secondary leading-relaxed">{ex}</div>
{/each}
</div>
</div>
{/if}
</div>
{/if}
</div>
{/each}
</div>
<!-- Add category button -->
<button
onclick={() => openAddModal()}
class="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-surface/50 px-4 py-3 text-sm text-text-muted transition-colors hover:border-interactive/50 hover:text-interactive"
>
<svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 4v16m8-8H4"/></svg>
Add category
</button>
{/if}
{/if}

<!-- Tab: Seeds -->
{#if activeTab === 'seeds'}
<!-- Sub-tab toggle -->
<div class="mb-4 flex justify-center">
<div class="flex items-center gap-1 rounded-lg bg-surface p-1">
	<button
		class="rounded-md px-3 py-1.5 text-xs font-medium transition-colors {seedsSubTab === 'query' ? 'bg-surface-2 text-text shadow-sm' : 'text-text-muted hover:text-text-secondary'}"
		onclick={() => seedsSubTab = 'query'}
	>
		Prompts <span class="ml-1 font-mono text-text-muted">{data.promptSeeds.length}</span>
	</button>
	<button
		class="rounded-md px-3 py-1.5 text-xs font-medium transition-colors {seedsSubTab === 'scenarios' ? 'bg-surface-2 text-text shadow-sm' : 'text-text-muted hover:text-text-secondary'}"
		onclick={() => seedsSubTab = 'scenarios'}
	>
		Scenarios <span class="ml-1 font-mono text-text-muted">{data.scenarioSeeds.length}</span>
	</button>
</div>
</div>

{#if seedsSubTab === 'query'}
{#if data.promptSeeds.length === 0}
<div class="rounded-lg border border-border bg-surface px-6 py-10 text-center">
<svg class="mx-auto mb-3 h-8 w-8 text-text-muted opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
<path d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"/>
</svg>
<p class="text-sm text-text-secondary">No prompt seeds generated yet.</p>
<p class="mt-1 text-xs text-text-muted">Run the pipeline to generate prompt seeds.</p>
</div>
{:else}
<div class="mb-3 flex flex-wrap items-center gap-3">
<div class="relative">
<svg class="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
</svg>
<input
type="text"
placeholder="Search seeds…"
bind:value={promptSeedFilter}
class="rounded-md border border-border bg-surface py-1.5 pl-8 pr-3 text-sm text-text placeholder-text-muted outline-none transition-colors focus:border-interactive focus:ring-1 focus:ring-interactive/50"
/>
</div>
<select
bind:value={promptSeedPermissibleFilter}
class="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text outline-none transition-colors focus:border-interactive"
>
<option value="all">All behaviors</option>
<option value="true">Permissible</option>
<option value="false">Not permissible</option>
</select>
<span class="text-xs text-text-muted">{filteredPromptSeeds.length} of {data.promptSeeds.length}</span>
</div>
<SeedGroupList
	groups={promptSeedsBySubRisk}
	expandedGroup={expandedPromptSeedSubRisk}
	onToggle={togglePromptSeedSubRisk}
/>
{/if}

{:else}
<!-- Scenarios sub-tab -->
{#if data.scenarioSeeds.length === 0}
<div class="rounded-lg border border-border bg-surface px-6 py-10 text-center">
<svg class="mx-auto mb-3 h-8 w-8 text-text-muted opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
<path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
</svg>
<p class="text-sm text-text-secondary">No audit scenarios generated yet.</p>
<p class="mt-1 text-xs text-text-muted">Generate adversarial multi-turn scenario seeds.</p>
</div>
{:else}
<div class="mb-3 flex flex-wrap items-center gap-3">
<div class="relative">
<svg class="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
</svg>
<input
type="text"
placeholder="Search seeds…"
bind:value={scenarioSeedFilter}
class="rounded-md border border-border bg-surface py-1.5 pl-8 pr-3 text-sm text-text placeholder-text-muted outline-none transition-colors focus:border-interactive focus:ring-1 focus:ring-interactive/50"
/>
</div>
<select
bind:value={scenarioSeedPermissibleFilter}
class="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text outline-none transition-colors focus:border-interactive"
>
<option value="all">All behaviors</option>
<option value="true">Permissible</option>
<option value="false">Not permissible</option>
</select>
<span class="text-xs text-text-muted">{filteredScenarioSeeds.length} of {data.scenarioSeeds.length}</span>
</div>
<SeedGroupList
	groups={scenarioSeedsBySubRisk}
	expandedGroup={expandedAuditSubRisk}
	onToggle={toggleAuditSubRisk}
/>
{/if}
{/if}
{/if}

<!-- Tab: Results -->
{#if activeTab === 'results'}

{#if allRuns.length === 0}
<div class="rounded-lg border border-border bg-surface px-6 py-10 text-center">
<svg class="mx-auto mb-3 h-8 w-8 text-text-muted opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
<path d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
</svg>
<p class="text-sm text-text-secondary">No evaluation runs found for this suite.</p>
<p class="mt-1 text-xs text-text-muted">Add results under <code>artifacts/results/{data.suite_id}</code> to browse them here.</p>
</div>
{:else if allRuns.length > 0}
<!-- Compare button (sticky) -->
{#if selectedRuns.size >= 1}
<div class="flex items-center gap-3 rounded-lg border border-border bg-surface px-4 py-2.5 mb-1">
	<span class="text-xs text-text-muted">{selectedRuns.size} selected</span>
	{#if canCompare}
		<a href="/suite/{data.suite_id}/compare?runs={[...selectedRuns].join(',')}"
			class="inline-flex items-center gap-1.5 rounded-md bg-interactive px-3 py-1 text-xs font-medium text-white hover:bg-interactive-hover transition-colors">
			Compare runs
			<svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6"/></svg>
		</a>
	{:else}
		<span class="text-[10px] text-text-muted">Select {2 - selectedRuns.size} more to compare</span>
	{/if}
	<button onclick={() => { selectedRuns = new Set(); }} class="ml-auto text-[10px] text-text-muted hover:text-interactive transition-colors">Clear</button>
</div>
{/if}

<div class="space-y-3">
{#each allRuns as run (run.run_id)}
{@const qRun = run.query}
{@const aRun = run.audit}
{@const qAvg = qRun?.metrics ? qRun.metrics.policy_violation_rate : 0}
{@const aAvg = aRun?.metrics ? aRun.metrics.policy_violation_rate : 0}
{@const bestAvg = Math.max(qAvg, aAvg)}
{@const isSelected = run.compare_run_id ? selectedRuns.has(run.compare_run_id) : false}
{@const compareDisabled = !run.compare_run_id}
<div class="rounded-lg border bg-surface overflow-hidden border-l-[3px] transition-colors duration-150 {isSelected ? 'border-interactive/50 border-l-interactive' : bestAvg >= 0.5 ? 'border-border border-l-score-fail' : bestAvg > 0 ? 'border-border border-l-score-border' : 'border-border border-l-score-pass'}">
<!-- Run header -->
<div class="flex items-center gap-3 px-4 py-2.5 border-b border-border/50">
<button onclick={() => toggleRunSelection(run.compare_run_id)}
	class="flex h-4 w-4 flex-shrink-0 items-center justify-center rounded border transition-colors duration-150 {compareDisabled ? 'cursor-not-allowed border-text-muted/20 opacity-40' : isSelected ? 'border-interactive bg-interactive' : 'border-text-muted/40 hover:border-interactive/60'}"
	title={compareDisabled ? 'Query results required for compare' : 'Select for comparison'}
	disabled={compareDisabled}>
	{#if isSelected}
		<svg class="h-3 w-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>
	{/if}
</button>
<div class="min-w-0">
<div class="truncate text-sm font-medium text-interactive">{run.run_id}</div>
<div class="truncate font-mono text-[10px] text-text-muted">{run.run_id}</div>
</div>
</div>

<!-- Query row -->
{#if qRun}
<a href="/suite/{data.suite_id}/{run.query_run_id ?? run.run_id}"
class="flex flex-wrap items-center gap-2 px-4 py-2.5 transition-colors hover:bg-surface/50 {aRun ? 'border-b border-border/30' : ''}">
<span class="w-20 flex-shrink-0 text-[10px] font-semibold uppercase tracking-wider text-text-muted">Prompts</span>
{#if qRun.metrics}
{@const total = qRun.metrics.total}
{@const avg = qRun.metrics.policy_violation_rate}
{@const judgeFailures = qRun.metrics.judge_failures}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px]">
<span class="text-text-muted">policy violation</span>
<span class="font-semibold tabular-nums {avg >= 0.5 ? 'text-score-fail' : avg > 0 ? 'text-score-border' : 'text-score-pass'}">{(avg * 100).toFixed(0)}%</span>
</span>
{#each Object.entries(qRun.metrics.dimensions ?? {}) as [dim, d]}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px]">
<span class="text-text-muted">{dim.replace(/_/g, ' ')}</span>
<span class="font-semibold tabular-nums {d.rate >= 0.5 ? 'text-score-fail' : d.rate > 0 ? 'text-score-border' : 'text-score-pass'}">{(d.rate * 100).toFixed(0)}%</span>
</span>
{/each}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px] text-text-muted">
target: <span class="font-mono">{qRun.metrics.target}</span>
</span>
{#if qRun.metrics.judge_model}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px] text-text-muted">
judge: <span class="font-mono">{qRun.metrics.judge_model}</span>
</span>
{/if}
{#if judgeFailures > 0}
<span class="inline-flex items-center gap-1 rounded border border-amber-500/25 bg-amber-500/8 px-2 py-0.5 text-[10px] text-amber-300">
{judgeFailures} judgment{judgeFailures === 1 ? '' : 's'} failed
</span>
{/if}
<span class="ml-auto text-[10px] text-text-muted">{total} prompts</span>
{/if}
</a>
{/if}

<!-- Scenarios row -->
{#if aRun}
<a href="/suite/{data.suite_id}/{run.audit_run_id ?? run.run_id}?tab=audit"
class="flex flex-wrap items-center gap-2 px-4 py-2.5 transition-colors hover:bg-surface/50">
<span class="w-20 flex-shrink-0 text-[10px] font-semibold uppercase tracking-wider text-text-muted">Scenarios</span>
{#if aRun.metrics}
{@const avg = aRun.metrics.policy_violation_rate}
{@const judgeFailures = aRun.metrics.judge_failures}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px]">
<span class="text-text-muted">policy violation</span>
<span class="font-semibold tabular-nums {avg >= 0.5 ? 'text-score-fail' : avg > 0 ? 'text-score-border' : 'text-score-pass'}">{(avg * 100).toFixed(0)}%</span>
</span>
{#each Object.entries(aRun.metrics.dimensions ?? {}) as [dim, d]}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px]">
<span class="text-text-muted">{dim.replace(/_/g, ' ')}</span>
<span class="font-semibold tabular-nums {d.rate >= 0.5 ? 'text-score-fail' : d.rate > 0 ? 'text-score-border' : 'text-score-pass'}">{(d.rate * 100).toFixed(0)}%</span>
</span>
{/each}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px] text-text-muted">
target: <span class="font-mono">{aRun.metrics.target}</span>
</span>
{#if aRun.metrics.auditor_model}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px] text-text-muted">
auditor: <span class="font-mono">{aRun.metrics.auditor_model}</span>
</span>
{/if}
{#if aRun.metrics.judge_model}
<span class="inline-flex items-center gap-1 rounded border border-border bg-background px-2 py-0.5 text-[10px] text-text-muted">
judge: <span class="font-mono">{aRun.metrics.judge_model}</span>
</span>
{/if}
{#if judgeFailures > 0}
<span class="inline-flex items-center gap-1 rounded border border-amber-500/25 bg-amber-500/8 px-2 py-0.5 text-[10px] text-amber-300">
{judgeFailures} judgment{judgeFailures === 1 ? '' : 's'} failed
</span>
{/if}
<span class="ml-auto text-[10px] text-text-muted">{aRun.metrics.total} scenarios</span>
{/if}
</a>
{/if}
</div>
{/each}
</div>
{/if}
{/if}
{/if}

<!-- Sub-risk Editor Modal -->
{#if editModalOpen}
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="fixed inset-0 z-50 flex items-center justify-center p-4" onkeydown={(e) => { if (e.key === 'Escape') closeEditModal(); }}>
<button class="absolute inset-0 bg-black/60 backdrop-blur-sm" onclick={() => closeEditModal()} aria-label="Close"></button>
<div class="relative w-full max-w-lg rounded-xl border border-border bg-bg shadow-2xl flex flex-col">
<div class="flex items-center justify-between border-b border-border px-6 py-4">
<h2 class="text-base font-semibold text-text">{editingIndex !== null ? 'Edit Category' : 'Add Category'}</h2>
<button onclick={() => closeEditModal()} aria-label="Close category editor" class="rounded-lg p-1.5 text-text-muted hover:text-text hover:bg-surface transition-colors">
<svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M6 18L18 6M6 6l12 12"/></svg>
</button>
</div>
<div class="px-6 py-5 space-y-4">
<div>
<label for="sr-name" class="block text-xs font-medium text-text-secondary mb-1">Name</label>
<input id="sr-name" type="text" bind:value={editForm.name} disabled={editingIndex !== null}
class="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-interactive {editingIndex !== null ? 'opacity-60' : ''}" />
</div>
<div>
<label for="sr-def" class="block text-xs font-medium text-text-secondary mb-1">Definition</label>
<textarea id="sr-def" bind:value={editForm.definition} rows={4}
class="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text leading-relaxed outline-none focus:border-interactive resize-y"></textarea>
</div>
<div>
<label for="sr-examples" class="block text-xs font-medium text-text-secondary mb-1">Examples <span class="text-text-muted font-normal">(one per line)</span></label>
<textarea id="sr-examples" bind:value={editExamplesText} rows={4}
class="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text leading-relaxed outline-none focus:border-interactive resize-y font-mono"></textarea>
</div>
<div class="flex items-center gap-3">
<span class="text-xs font-medium text-text-secondary">Behavior</span>
<button
onclick={() => editForm.permissible = !editForm.permissible}
class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors {editForm.permissible ? 'bg-interactive/10 text-interactive border border-interactive/30' : 'bg-not-permissible/10 text-not-permissible border border-not-permissible/30'}"
>
{editForm.permissible ? 'permissible' : 'not permissible'}
</button>
</div>
{#if editError}
<p class="text-xs text-not-permissible">{editError}</p>
{/if}
</div>
<div class="flex items-center justify-end gap-3 border-t border-border px-6 py-4">
<button onclick={() => closeEditModal()} class="rounded-md px-4 py-2 text-sm text-text-muted hover:text-text transition-colors">Cancel</button>
<button onclick={() => handleSaveSubRisk()} disabled={editSaving}
class="rounded-md bg-interactive px-4 py-2 text-sm font-medium text-white hover:bg-interactive-hover transition-colors disabled:opacity-50">
{editSaving ? 'Saving…' : editingIndex !== null ? 'Save changes' : 'Add category'}
</button>
</div>
</div>
</div>
{/if}

<!-- Delete Confirmation -->
{#if deleteConfirmIndex !== null}
<div class="fixed inset-0 z-50 flex items-center justify-center p-4">
<button class="absolute inset-0 bg-black/60 backdrop-blur-sm" onclick={() => deleteConfirmIndex = null} aria-label="Close"></button>
<div class="relative w-full max-w-sm rounded-xl border border-border bg-bg shadow-2xl p-6 text-center">
<svg class="mx-auto mb-3 h-10 w-10 text-not-permissible/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
<h3 class="text-sm font-semibold text-text">Delete "{sortedSubRisks[deleteConfirmIndex]?.name}"?</h3>
<p class="mt-1.5 text-xs text-text-muted">This category will be removed from the policy. This cannot be undone.</p>
<div class="mt-5 flex justify-center gap-3">
<button onclick={() => deleteConfirmIndex = null} class="rounded-md px-4 py-2 text-sm text-text-muted hover:text-text transition-colors">Cancel</button>
<button onclick={() => { if (deleteConfirmIndex !== null) handleDeleteSubRisk(deleteConfirmIndex); }} class="rounded-md bg-not-permissible px-4 py-2 text-sm font-medium text-white hover:bg-not-permissible/80 transition-colors">Delete</button>
</div>
</div>
</div>
{/if}

<!-- Seeds Warning Modal -->
{#if seedsWarningPending}
<div class="fixed inset-0 z-50 flex items-center justify-center p-4">
<button class="absolute inset-0 bg-black/60 backdrop-blur-sm" onclick={() => { seedsWarningPending = false; pendingPolicy = null; }} aria-label="Close"></button>
<div class="relative w-full max-w-md rounded-xl border border-border bg-bg shadow-2xl p-6">
<div class="flex items-start gap-3">
<div class="flex-shrink-0 mt-0.5 flex h-8 w-8 items-center justify-center rounded-full bg-yellow-500/10">
<svg class="h-4 w-4 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"/></svg>
</div>
<div>
<h3 class="text-sm font-semibold text-text">Existing seeds won't be updated</h3>
<p class="mt-1.5 text-xs text-text-muted leading-relaxed">
This suite has <strong class="text-text-secondary">{data.promptSeeds.length} prompts</strong>
{#if data.scenarioSeeds.length > 0}
and <strong class="text-text-secondary">{data.scenarioSeeds.length} scenarios</strong>
{/if}
that were generated from the previous policy. Editing the policy won't update them — you'll need to regenerate seeds for changes to take effect.
</p>
</div>
</div>
<div class="mt-5 flex justify-end gap-3">
<button onclick={() => { seedsWarningPending = false; pendingPolicy = null; }} class="rounded-md px-4 py-2 text-sm text-text-muted hover:text-text transition-colors">Cancel</button>
<button onclick={() => confirmSaveWithSeeds()} disabled={editSaving}
class="rounded-md bg-interactive px-4 py-2 text-sm font-medium text-white hover:bg-interactive-hover transition-colors disabled:opacity-50">
{editSaving ? 'Saving…' : 'Save anyway'}
</button>
</div>
</div>
</div>
{/if}



<!-- Systematization Modal -->
<SystematizationModal
	bind:open={systematizationModalOpen}
	systematization={data.systematization as Record<string, unknown> | null}
/>
