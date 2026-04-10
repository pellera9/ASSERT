<script lang="ts">
	import { renderMarkdown } from '$lib/markdown';
	import type { SeedTool, ViewerSeedGroup } from '$lib/types.js';
	import { slide } from 'svelte/transition';
	import { quintOut } from 'svelte/easing';

	let {
		groups,
		expandedGroup,
		onToggle
	}: {
		groups: ViewerSeedGroup[];
		expandedGroup: string | null;
		onToggle: (name: string) => void;
	} = $props();

	let expandedTools = $state<Set<string>>(new Set());

	function toggleTools(id: string) {
		const next = new Set(expandedTools);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		expandedTools = next;
	}

	function toolNames(tools: SeedTool[] | undefined): string {
		return tools?.map((tool) => tool.name).join(', ') ?? '';
	}
</script>

<div class="overflow-hidden rounded-lg border border-border">
	{#each groups as group, gIdx (group.name)}
		<div class={gIdx > 0 ? 'border-t border-border' : ''}>
			<button
				class="flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors hover:bg-surface {expandedGroup === group.name ? 'bg-surface' : ''}"
				onclick={() => onToggle(group.name)}
			>
				<span class="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium {group.permissible ? 'bg-interactive/10 text-interactive' : 'bg-not-permissible/10 text-not-permissible'}">
					{group.permissible ? 'permissible' : 'not permissible'}
				</span>
				<span class="flex-1 truncate font-medium">{group.name}</span>
				<span class="rounded bg-surface-2 px-2 py-0.5 text-xs font-mono text-text-muted">{group.items.length}</span>
				<svg class="h-3.5 w-3.5 text-text-muted transition-transform duration-200 {expandedGroup === group.name ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
					<path d="M9 5l7 7-7 7"/>
				</svg>
			</button>
			{#if expandedGroup === group.name}
				<div class="border-t border-border" transition:slide={{ duration: 200, easing: quintOut }}>
					{#if group.definition}
						<div class="border-b border-border bg-surface px-5 py-4">
							<div class="prose text-sm text-text-secondary leading-relaxed">{@html renderMarkdown(group.definition)}</div>
						</div>
					{/if}
					<div class="divide-y divide-border">
						{#each group.items as entry, sIdx (entry.id)}
							{@const toolsOpen = expandedTools.has(entry.id)}
							<div class="px-5 py-4">
								<div class="flex items-start gap-3">
									<span class="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-surface-2 text-[11px] font-mono text-text-muted">{sIdx + 1}</span>
									<div class="min-w-0 flex-1">
										<div class="mb-2 flex items-center gap-2">
											<h5 class="text-sm font-semibold text-text">{entry.title}</h5>
											{#if entry.elicitation_strategy}
												<span class="flex-shrink-0 rounded border border-violet-500/20 bg-violet-500/10 px-1.5 py-0.5 text-[10px] font-medium text-violet-400">{entry.elicitation_strategy}</span>
											{/if}
										</div>
										<div class="prose text-sm text-text-secondary leading-relaxed">{@html renderMarkdown(entry.description)}</div>
										{#if entry.system_prompt}
											<div class="mt-3 rounded border border-yellow-500/20 bg-yellow-500/5 px-3 py-2">
												<div class="mb-1 text-[10px] font-semibold uppercase tracking-wider text-yellow-400">System prompt</div>
												<div class="text-xs text-text-secondary leading-relaxed">{@html renderMarkdown(entry.system_prompt)}</div>
											</div>
										{/if}
										{#if entry.tools && entry.tools.length > 0}
											<div class="mt-3">
												<button class="group flex items-center gap-1.5" onclick={() => toggleTools(entry.id)}>
													<svg class="h-3 w-3 text-purple-400/60 transition-transform duration-150 {toolsOpen ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3"><path d="M9 5l7 7-7 7"/></svg>
													<span class="text-[11px] font-semibold uppercase tracking-wider text-purple-400">Tools</span>
													<span class="text-[10px] text-text-muted">{toolNames(entry.tools)}</span>
												</button>
												{#if toolsOpen}
													<div class="mt-1.5 space-y-2">
														{#each entry.tools as tool}
															<div class="rounded border border-purple-500/20 bg-purple-500/5 px-3 py-2">
																<span class="font-mono text-xs font-semibold text-purple-300">{tool.name}</span>
																{#if tool.description}
																	<p class="mt-0.5 text-xs text-text-secondary">{tool.description}</p>
																{/if}
																{#if tool.parameters?.length}
																	<div class="mt-1.5 space-y-0.5">
																		{#each tool.parameters as parameter}
																			<div class="flex items-baseline gap-1.5">
																				<span class="inline-flex items-center rounded border border-border/20 bg-surface px-1.5 py-0.5 font-mono text-[10px] text-text-muted">
																					{parameter.name}<span class="ml-0.5 text-text-muted/50">:{parameter.type || 'string'}</span>
																				</span>
																				{#if parameter.description}
																					<span class="text-[10px] text-text-muted/70">{parameter.description}</span>
																				{/if}
																			</div>
																		{/each}
																	</div>
																{/if}
															</div>
														{/each}
													</div>
												{/if}
											</div>
										{/if}
									</div>
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	{/each}
</div>
