import type {
	AuditScore,
	AuditTranscriptMessage,
	InteractionMessage,
	JudgedSample,
	LlmCallTrace,
	ScenarioSeedInfo,
	ViewerResultItem
} from '$lib/types.js';

function normalizeMessageRole(role: string): InteractionMessage['role'] {
	if (role === 'system' || role === 'user' || role === 'assistant' || role === 'tool') {
		return role;
	}
	return 'assistant';
}

function normalizeInteractionMessages(messages: InteractionMessage[]): InteractionMessage[] {
	let judgeTurn = 0;
	return messages.map((message) => {
		const role = normalizeMessageRole(message.role);
		const nextJudgeTurn = role === 'system' ? null : judgeTurn + 1;
		if (nextJudgeTurn != null) judgeTurn = nextJudgeTurn;
		return {
			...message,
			role,
			judgeTurn: nextJudgeTurn
		};
	});
}

function toInteractionMessages(messages: AuditTranscriptMessage[]): InteractionMessage[] {
	return normalizeInteractionMessages(messages.map((message) => ({
		id: message.id,
		role: normalizeMessageRole(message.role),
		content: message.content,
		type: message.type,
		judgeTurn: message.judgeTurn,
		tool_calls: message.tool_calls,
		tool_call_id: message.tool_call_id,
		function: message.function,
		arguments: message.arguments,
		raw: message.raw
	})));
}

function readSeedString(seedMetadata: Record<string, unknown> | null | undefined, key: string): string | null {
	const value = seedMetadata?.[key];
	return typeof value === 'string' && value.trim() ? value : null;
}

function readSeedTools(seedMetadata: Record<string, unknown> | null | undefined) {
	return Array.isArray(seedMetadata?.tools) ? seedMetadata.tools : undefined;
}

function countConversationMessages(messages: InteractionMessage[]): number {
	return messages.filter((message) => message.role !== 'system').length;
}

function synthesizePromptMessages(sample: JudgedSample): InteractionMessage[] {
	const messages: InteractionMessage[] = [];
	let judgeTurn = 1;
	const systemPrompt = readSeedString(sample.seed_metadata, 'system_prompt');

	if (systemPrompt) {
		messages.push({
			id: 'prompt:system',
			role: 'system',
			content: systemPrompt,
			type: 'set_system_message',
			judgeTurn: null
		});
	}

	messages.push({
		id: 'prompt:user',
		role: 'user',
		content: sample.prompt,
		type: 'message',
		judgeTurn
	});
	judgeTurn += 1;

	messages.push({
		id: 'prompt:assistant',
		role: 'assistant',
		content: sample.response,
		type: 'message',
		judgeTurn
	});

	return messages;
}

export function normalizePromptResult(sample: JudgedSample): ViewerResultItem {
	const messages = normalizeInteractionMessages(
		sample.messages && sample.messages.length > 0 ? sample.messages : synthesizePromptMessages(sample)
	);

	return {
		id: `prompt:${sample.run_id ?? 'run'}:${sample.sub_risk}:${sample.prompt.slice(0, 80)}`,
		kind: 'prompt',
		row_title: sample.prompt,
		header_title: sample.sub_risk,
		sub_risk: sample.sub_risk,
		permissible: sample.permissible ?? true,
		verdict: sample.verdict,
		judge_status: sample.judge_status,
		judge_error: sample.judge_error,
		multi_judge: sample.multi_judge,
		messages,
		llm_calls: sample.llm_calls ?? [],
		target_runtime_mode: sample.target_runtime_mode ?? null,
		context: {
			tools: readSeedTools(sample.seed_metadata),
			turns_count: countConversationMessages(messages)
		}
	};
}

export function normalizeScenarioResult(
	score: AuditScore,
	messages: AuditTranscriptMessage[],
	llmCalls: LlmCallTrace[],
	seedInfo: ScenarioSeedInfo | undefined,
	scenarioSeedMap: Record<string, ScenarioSeedInfo>
): ViewerResultItem {
	const interactionMessages = toInteractionMessages(messages);
	const parentTitle =
		seedInfo?.parent_seed_id && scenarioSeedMap[seedInfo.parent_seed_id]
			? scenarioSeedMap[seedInfo.parent_seed_id].title
			: null;

	return {
		id: `scenario:${score.seed_id}`,
		kind: 'scenario',
		row_title: seedInfo?.title ?? score.seed_id,
		header_title: seedInfo?.title ?? score.seed_id,
		sub_risk: score.sub_risk,
		permissible: score.permissible,
		verdict: score.verdict,
		judge_status: score.judge_status,
		judge_error: score.judge_error,
		multi_judge: score.multi_judge,
		messages: interactionMessages,
		llm_calls: llmCalls,
		target_runtime_mode: score.target_runtime_mode ?? null,
		context: {
			description: seedInfo?.description ?? null,
			tools: seedInfo?.tools,
			turns_count: countConversationMessages(interactionMessages),
			stop_reason: score.metadata.stop_reason,
			elicitation_strategy: seedInfo?.elicitation_strategy ?? null,
			parent_title: parentTitle
		}
	};
}
