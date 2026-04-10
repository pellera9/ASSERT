import { json } from '@sveltejs/kit';
import { loadDimensions, saveDimension } from '$lib/server/dimensions.js';
import type { RequestHandler } from './$types.js';

export const GET: RequestHandler = async () => {
	return json(loadDimensions());
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json();
	const { name, description, rubric } = body;

	if (!name || !description || !rubric) {
		return json({ error: 'name, description, and rubric are required' }, { status: 400 });
	}

	const nameClean = name.trim().toLowerCase().replace(/[^a-z0-9_]/g, '_');
	if (!nameClean) {
		return json({ error: 'Invalid dimension name' }, { status: 400 });
	}

	const existing = loadDimensions();
	if (nameClean in existing) {
		return json({ error: `Dimension "${nameClean}" already exists` }, { status: 409 });
	}

	saveDimension(nameClean, description.trim(), rubric.trim());
	return json({ name: nameClean }, { status: 201 });
};
