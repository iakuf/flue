import { describe, it, expect, vi } from 'vitest';
import { parallel, pipeline, phase, log, registerWorkflow } from '../src/orchestrate.ts';
import type { FlueHarness, FlueSession, PromptResponse } from '../src/types.ts';

/**
 * A mock session that faithfully reproduces Flue's exclusive-operation lock:
 * starting a second operation while one is active throws, exactly like the
 * real runtime. This is what makes the concurrency tests meaningful.
 */
function createLockedSession(
	name: string,
	behavior: (prompt: string) => Promise<PromptResponse>,
): FlueSession {
	let active = false;
	return {
		name,
		async prompt(prompt: string): Promise<PromptResponse> {
			if (active) {
				throw new Error(
					`[flue] Session "${name}" is already running prompt. Start another session for parallel conversation branches.`,
				);
			}
			active = true;
			try {
				return await behavior(prompt);
			} finally {
				active = false;
			}
		},
	} as unknown as FlueSession;
}

/**
 * A mock harness that hands out a fresh locked session per name and tracks
 * how many distinct sessions were created/deleted (to prove true isolation
 * and cleanup).
 */
function createMockHarness(
	behavior: (prompt: string) => Promise<PromptResponse> = async (p) =>
		({ text: `response-for-${p.slice(0, 20)}` }) as PromptResponse,
) {
	const created = new Set<string>();
	const deleted = new Set<string>();
	const sessions = new Map<string, FlueSession>();

	const harness = {
		name: 'mock',
		async session(name = 'default'): Promise<FlueSession> {
			created.add(name);
			let s = sessions.get(name);
			if (!s) {
				s = createLockedSession(name, behavior);
				sessions.set(name, s);
			}
			return s;
		},
		sessions: {
			async create(name = 'default') {
				return harness.session(name);
			},
			async get(name = 'default') {
				return harness.session(name);
			},
			async delete(name = 'default') {
				deleted.add(name);
				sessions.delete(name);
			},
		},
	} as unknown as FlueHarness;

	return { harness, created, deleted };
}

describe('parallel()', () => {
	it('runs multiple tasks and returns all results in order', async () => {
		const { harness } = createMockHarness(async (prompt) => {
			if (prompt.includes('auth')) return { text: 'auth-result' } as PromptResponse;
			if (prompt.includes('rate')) return { text: 'rate-result' } as PromptResponse;
			return { text: 'cache-result' } as PromptResponse;
		});

		const results = await parallel(harness, [
			{ prompt: 'Analyze auth' },
			{ prompt: 'Analyze rate' },
			{ prompt: 'Analyze cache' },
		]);

		expect(results).toHaveLength(3);
		expect(results[0]?.text).toBe('auth-result');
		expect(results[1]?.text).toBe('rate-result');
		expect(results[2]?.text).toBe('cache-result');
	});

	it('uses a DISTINCT session per task (true isolation, no lock contention)', async () => {
		// Each task sleeps so they overlap in time. If they shared one session,
		// the locked-session mock would throw. Distinct sessions => all succeed.
		const { harness, created, deleted } = createMockHarness(async (p) => {
			await new Promise((r) => setTimeout(r, 30));
			return { text: `done-${p}` } as PromptResponse;
		});

		const results = await parallel(
			harness,
			Array.from({ length: 5 }, (_, i) => ({ prompt: `task-${i}` })),
		);

		expect(results.every((r) => r !== null)).toBe(true);
		expect(created.size).toBe(5);
		expect(deleted.size).toBe(5);
	});

	it('respects concurrency limit', async () => {
		let concurrent = 0;
		let maxConcurrent = 0;
		const { harness } = createMockHarness(async () => {
			concurrent++;
			maxConcurrent = Math.max(maxConcurrent, concurrent);
			await new Promise((r) => setTimeout(r, 40));
			concurrent--;
			return { text: 'done' } as PromptResponse;
		});

		await parallel(
			harness,
			Array.from({ length: 10 }, (_, i) => ({ prompt: `t-${i}` })),
			{ concurrency: 3 },
		);

		expect(maxConcurrent).toBeLessThanOrEqual(3);
	});

	it('returns empty array for empty input', async () => {
		const { harness } = createMockHarness();
		expect(await parallel(harness, [])).toEqual([]);
	});

	it('lenient mode: failed tasks return null, others complete', async () => {
		const { harness } = createMockHarness(async (p) => {
			if (p.includes('bad')) throw new Error('boom');
			return { text: `ok-${p}` } as PromptResponse;
		});

		const results = await parallel(harness, [
			{ prompt: 'good 1' },
			{ prompt: 'bad' },
			{ prompt: 'good 2' },
		]);

		expect(results[0]?.text).toContain('ok-');
		expect(results[1]).toBeNull();
		expect(results[2]?.text).toContain('ok-');
	});

	it('strict mode: first failure aborts and throws', async () => {
		const { harness } = createMockHarness(async (p) => {
			if (p.includes('fail')) throw new Error('Task exploded');
			await new Promise((r) => setTimeout(r, 10));
			return { text: 'ok' } as PromptResponse;
		});

		await expect(
			parallel(
				harness,
				[{ prompt: 'ok' }, { prompt: 'fail' }, { prompt: 'ok' }],
				{ failMode: 'strict' },
			),
		).rejects.toThrow('Task exploded');
	});

	it('cleans up sessions even when a task fails', async () => {
		const { harness, created, deleted } = createMockHarness(async (p) => {
			if (p.includes('bad')) throw new Error('boom');
			return { text: 'ok' } as PromptResponse;
		});

		await parallel(harness, [{ prompt: 'good' }, { prompt: 'bad' }]);

		expect(deleted.size).toBe(created.size);
		expect(deleted.size).toBe(2);
	});

	it('throws if signal already aborted', async () => {
		const { harness } = createMockHarness();
		const controller = new AbortController();
		controller.abort();
		await expect(
			parallel(harness, [{ prompt: 'x' }], { signal: controller.signal }),
		).rejects.toThrow('Aborted');
	});
});

describe('pipeline()', () => {
	it('runs stages sequentially within an item, passing output forward', async () => {
		const seen: string[] = [];
		const { harness } = createMockHarness(async (p) => {
			seen.push(p);
			return { text: `${p}=>out` } as PromptResponse;
		});

		const results = await pipeline(harness, [{ text: 'seed' } as PromptResponse], [
			(input) => ({ prompt: `S1:${input.text}` }),
			(input) => ({ prompt: `S2:${input.text}` }),
			(input) => ({ prompt: `S3:${input.text}` }),
		]);

		expect(results).toHaveLength(1);
		expect(seen[0]).toBe('S1:seed');
		expect(seen[1]).toBe('S2:S1:seed=>out');
		expect(seen[2]).toContain('S3:');
	});

	it('reuses ONE session across an item stages (sequential, no lock error)', async () => {
		const { harness, created, deleted } = createMockHarness(async () => {
			await new Promise((r) => setTimeout(r, 10));
			return { text: 'ok' } as PromptResponse;
		});

		await pipeline(harness, [{ text: 'a' } as PromptResponse], [
			(i) => ({ prompt: `1:${i.text}` }),
			(i) => ({ prompt: `2:${i.text}` }),
		]);

		expect(created.size).toBe(1);
		expect(deleted.size).toBe(1);
	});

	it('processes multiple items on distinct sessions', async () => {
		const { harness, created } = createMockHarness();
		const items = [
			{ text: 'i1' } as PromptResponse,
			{ text: 'i2' } as PromptResponse,
			{ text: 'i3' } as PromptResponse,
		];

		const results = await pipeline(harness, items, [(i) => ({ prompt: `go:${i.text}` })]);

		expect(results).toHaveLength(3);
		expect(created.size).toBe(3);
	});

	it('returns empty for empty items or stages', async () => {
		const { harness } = createMockHarness();
		expect(await pipeline(harness, [], [(i) => ({ prompt: i.text })])).toEqual([]);
		expect(await pipeline(harness, [{ text: 'x' } as PromptResponse], [])).toEqual([]);
	});

	it('failed items return null without crashing others', async () => {
		const { harness } = createMockHarness(async (p) => {
			if (p.includes('bad')) throw new Error('boom');
			return { text: 'ok' } as PromptResponse;
		});

		const results = await pipeline(
			harness,
			[
				{ text: 'good' } as PromptResponse,
				{ text: 'bad' } as PromptResponse,
				{ text: 'good' } as PromptResponse,
			],
			[(i) => ({ prompt: `p:${i.text}` })],
		);

		expect(results[0]).not.toBeNull();
		expect(results[1]).toBeNull();
		expect(results[2]).not.toBeNull();
	});

	it('supports named workflows as string stages', async () => {
		const { harness } = createMockHarness();
		registerWorkflow('analyze', async (input, session) =>
			session.prompt(`analyze:${input.text}`),
		);

		const results = await pipeline(harness, [{ text: 'ticket' } as PromptResponse], [
			'analyze',
			(input) => ({ prompt: `summarize:${input.text}` }),
		]);

		expect(results).toHaveLength(1);
		expect(results[0]?.text).toBeDefined();
	});

	it('throws for unknown workflow name (fail fast, before execution)', async () => {
		const { harness } = createMockHarness();
		await expect(
			pipeline(harness, [{ text: 'x' } as PromptResponse], ['does-not-exist']),
		).rejects.toThrow('Unknown workflow');
	});
});

describe('phase() and log()', () => {
	it('phase() outputs to console', () => {
		const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
		phase('Research');
		expect(spy).toHaveBeenCalledWith(expect.stringContaining('Research'));
		spy.mockRestore();
	});

	it('log() outputs to console', () => {
		const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
		log('Found 5 items');
		expect(spy).toHaveBeenCalledWith(expect.stringContaining('Found 5 items'));
		spy.mockRestore();
	});
});
