import { loadSuitePageData } from '$lib/server/data.js';
import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types.js';

export const load: PageServerLoad = async ({ params }) => {
	const payload = loadSuitePageData(params.suite_id);
	if (!payload) throw error(404, `Suite "${params.suite_id}" not found`);
	return payload;
};
