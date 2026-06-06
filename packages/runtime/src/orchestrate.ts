/**
 * Deterministic multi-agent orchestration primitives for Flue.
 *
 * Inspired by Claude Code Dynamic Workflows. These primitives give you
 * code-driven control over "what runs when" while letting the LLM handle
 * "what to say/do" within each step.
 *
 * ## Concurrency model
 *
 * A Flue {@link FlueSession} runs operations **exclusively** — starting a
 * second operation on a session that is already running one throws
 * ("Start another session for parallel conversation branches"). True
 * parallelism therefore requires **one session per concurrent unit**.
 *
 * These primitives take a {@link FlueHarness} (not a single session) and
 * allocate an isolated session for each concurrent task, then clean it up
 * when done. Callers describe tasks; the orchestrator owns session lifecycle.
 *
 * Design principles:
 * - Correct: never runs two operations on one session (respects the runtime lock)
 * - Deterministic: orchestration logic is TypeScript, not LLM-generated
 * - Barrier-aware: parallel() waits for all; pipeline() is barrier-free
 * - Fault-tolerant: failed tasks return null by default, never crash the batch
 * - Self-cleaning: isolated sessions are deleted after use
 *
 * @example
 * ```ts
 * import { parallel, pipeline, phase } from './orchestrate.ts';
 *
 * const harness = await init(agent);
 *
 * phase('Research');
 * const results = await parallel(harness, [
 *   { prompt: 'Analyze auth flow', label: 'auth' },
 *   { prompt: 'Check rate limiting', label: 'rate-limit' },
 * ]);
 * ```
 *
 * @module
 */

import type {
	FlueHarness,
	FlueSession,
	PromptResponse,
	PromptOptions,
} from './types.ts';
import type * as v from 'valibot';

// ─── Types ──────────────────────────────────────────────────────────────────

/** A single task descriptor for parallel/pipeline execution. */
export interface TaskDescriptor<S extends v.GenericSchema | undefined = undefined> {
	/** The prompt to send to the agent. */
	prompt: string;
	/** Display label for observability (like Claude Code's `label` option). */
	label?: string;
	/** Optional valibot schema for structured result extraction. */
	result?: S;
	/** Override model for this specific task. */
	model?: string;
}

export interface ParallelOptions {
	/**
	 * Maximum number of tasks running concurrently.
	 * Default: 16 (matches Claude Code Workflows limit).
	 */
	concurrency?: number;
	/**
	 * How to handle individual task failures:
	 * - "lenient": failed tasks return null, others continue (default)
	 * - "strict": first failure aborts all remaining tasks
	 */
	failMode?: 'lenient' | 'strict';
	/** AbortSignal to cancel all pending tasks. */
	signal?: AbortSignal;
	/**
	 * Prefix for the isolated session names this call allocates.
	 * Default: "parallel". Useful for tracing/debugging.
	 */
	sessionPrefix?: string;
}

export interface PipelineOptions {
	/**
	 * Maximum items processed concurrently through the full stage sequence.
	 * Default: 16.
	 */
	concurrency?: number;
	/** AbortSignal to cancel processing. */
	signal?: AbortSignal;
	/** Prefix for the isolated session names. Default: "pipeline". */
	sessionPrefix?: string;
}

/** Result from a parallel() or pipeline() execution. */
export type OrchestrationResult = PromptResponse | null;

// ─── Phase / log ────────────────────────────────────────────────────────────

/**
 * Mark a new orchestration phase. Phases are logical groupings for
 * observability. Equivalent to Claude Code Workflows' `phase(title)`.
 */
export function phase(title: string): void {
	const timestamp = new Date().toISOString();
	console.log(`[orchestrate] ---- ${title} ---- (${timestamp})`);
}

/** Log a narrative message within the current orchestration flow. */
export function log(message: string): void {
	console.log(`[orchestrate] ${message}`);
}

// ─── Internal: run one task on a fresh isolated session ───────────────────────

async function runIsolated(
	harness: FlueHarness,
	sessionName: string,
	prompt: string,
	promptOpts: PromptOptions,
): Promise<PromptResponse> {
	const session = await harness.session(sessionName);
	try {
		return await session.prompt(prompt, promptOpts);
	} finally {
		// Best-effort cleanup. delete() rejects if an operation is still active,
		// which cannot happen here because prompt() has already settled.
		await harness.sessions.delete(sessionName).catch(() => {});
	}
}

function buildPromptOptions(
	task: Pick<TaskDescriptor, 'result' | 'model'>,
	signal?: AbortSignal,
): PromptOptions {
	const opts: PromptOptions = {};
	if (signal) opts.signal = signal;
	if (task.model) opts.model = task.model;
	if (task.result) (opts as any).result = task.result;
	return opts;
}

// ─── Parallel ───────────────────────────────────────────────────────────────

/**
 * Execute multiple tasks concurrently, each on its own isolated session.
 * Waits for ALL tasks to complete (or fail) before returning (barrier).
 *
 * Because each task runs on a dedicated session, this achieves true
 * parallelism without hitting the per-session exclusive-operation lock.
 *
 * Failed tasks return `null` in lenient mode (default). In strict mode the
 * first failure aborts remaining tasks and throws.
 *
 * @example
 * ```ts
 * const results = await parallel(harness, [
 *   { prompt: 'Research authentication patterns' },
 *   { prompt: 'Research rate limiting patterns' },
 *   { prompt: 'Research caching strategies' },
 * ], { concurrency: 8 });
 * ```
 */
export async function parallel(
	harness: FlueHarness,
	tasks: TaskDescriptor[],
	options?: ParallelOptions,
): Promise<OrchestrationResult[]> {
	const {
		concurrency = 16,
		failMode = 'lenient',
		signal,
		sessionPrefix = 'parallel',
	} = options ?? {};

	if (tasks.length === 0) return [];

	const results: OrchestrationResult[] = new Array(tasks.length).fill(null);
	const queue: { task: TaskDescriptor; index: number }[] = tasks.map((task, index) => ({
		task,
		index,
	}));

	const semaphore = new Semaphore(Math.min(concurrency, tasks.length));
	const abortController = new AbortController();
	let firstError: Error | null = null;

	if (signal) {
		if (signal.aborted) throw new Error('[orchestrate] Aborted before start');
		signal.addEventListener('abort', () => abortController.abort(signal.reason), { once: true });
	}

	const promises = queue.map(async ({ task, index }) => {
		await semaphore.acquire();
		if (abortController.signal.aborted) {
			semaphore.release();
			return;
		}

		const sessionName = `${sessionPrefix}-${index}-${crypto.randomUUID()}`;
		try {
			const promptOpts = buildPromptOptions(task, abortController.signal);
			results[index] = await runIsolated(harness, sessionName, task.prompt, promptOpts);
			if (task.label) log(`[OK] ${task.label}`);
		} catch (err) {
			if (failMode === 'strict' && !firstError) {
				firstError = err instanceof Error ? err : new Error(String(err));
				abortController.abort(firstError);
			}
			results[index] = null;
			if (task.label) {
				log(`[FAIL] ${task.label}: ${err instanceof Error ? err.message : err}`);
			}
		} finally {
			semaphore.release();
		}
	});

	await Promise.all(promises);

	if (failMode === 'strict' && firstError) throw firstError;

	return results;
}

// ─── Workflow Registry ──────────────────────────────────────────────────────

/**
 * A named, reusable workflow that takes the output of the previous stage and
 * returns the input for the next. It receives a dedicated isolated session.
 */
export type NamedWorkflow = (
	input: PromptResponse,
	session: FlueSession,
) => Promise<PromptResponse>;

const workflowRegistry = new Map<string, NamedWorkflow>();

/**
 * Register a named workflow for later reference in pipeline stages.
 *
 * @example
 * ```ts
 * registerWorkflow('analyze-ticket', async (input, session) =>
 *   session.prompt(`Analyze: ${input.text}`),
 * );
 * pipeline(harness, tickets, ['analyze-ticket', 'verify']);
 * ```
 */
export function registerWorkflow(name: string, workflow: NamedWorkflow): void {
	workflowRegistry.set(name, workflow);
}

/** Resolve a registered workflow by name. Throws if not found. */
export function resolveWorkflow(name: string): NamedWorkflow {
	const workflow = workflowRegistry.get(name);
	if (!workflow) {
		const registered = [...workflowRegistry.keys()].join(', ');
		throw new Error(
			`[orchestrate] Unknown workflow: "${name}". Registered: [${registered}]`,
		);
	}
	return workflow;
}

/** Returns all registered workflow names. */
export function listWorkflows(): string[] {
	return [...workflowRegistry.keys()];
}

// ─── Pipeline ───────────────────────────────────────────────────────────────

/**
 * A stage that takes the previous result and returns the next task descriptor.
 * Pass a string to invoke a named workflow registered via {@link registerWorkflow}.
 */
export type PipelineStage =
	| ((input: PromptResponse) => TaskDescriptor)
	| string;

/**
 * Process items through a multi-stage pipeline WITHOUT barrier.
 * Each item flows independently through all stages on its own isolated
 * session — fast items don't wait for slow ones.
 *
 * Within one item the stages run sequentially (each stage sees the previous
 * stage's output). Across items there is no synchronization.
 *
 * @example
 * ```ts
 * const responses = await pipeline(harness, tickets, [
 *   (ticket) => ({ prompt: `Analyze: ${ticket.text}` }),
 *   (analysis) => ({ prompt: `Verify: ${analysis.text}` }),
 *   'format-response',
 * ]);
 * ```
 */
export async function pipeline(
	harness: FlueHarness,
	items: PromptResponse[],
	stages: PipelineStage[],
	options?: PipelineOptions,
): Promise<OrchestrationResult[]> {
	const { concurrency = 16, signal, sessionPrefix = 'pipeline' } = options ?? {};

	if (items.length === 0 || stages.length === 0) return [];
	if (signal?.aborted) throw new Error('[orchestrate] Aborted before start');

	// Eagerly validate all named workflow references so bad names fail fast.
	for (const stage of stages) {
		if (typeof stage === 'string') resolveWorkflow(stage);
	}

	const semaphore = new Semaphore(Math.min(concurrency, items.length));

	return Promise.all(
		items.map(async (item, index) => {
			await semaphore.acquire();
			// One session per item, reused across this item's sequential stages.
			const sessionName = `${sessionPrefix}-${index}-${crypto.randomUUID()}`;
			let session: FlueSession | undefined;
			try {
				session = await harness.session(sessionName);
				let current: PromptResponse = item;

				for (const stage of stages) {
					if (signal?.aborted) return null;

					if (typeof stage === 'string') {
						const workflow = resolveWorkflow(stage);
						current = await workflow(current, session);
						continue;
					}

					const descriptor = stage(current);
					const promptOpts = buildPromptOptions(descriptor, signal);
					current = await session.prompt(descriptor.prompt, promptOpts);
				}

				return current;
			} catch {
				return null;
			} finally {
				if (session) {
					await harness.sessions.delete(sessionName).catch(() => {});
				}
				semaphore.release();
			}
		}),
	);
}

// ─── Semaphore ──────────────────────────────────────────────────────────────

/** Async semaphore for bounding concurrency. */
class Semaphore {
	private waiting: (() => void)[] = [];
	private count: number;

	constructor(max: number) {
		this.count = max;
	}

	async acquire(): Promise<void> {
		if (this.count > 0) {
			this.count--;
			return;
		}
		return new Promise<void>((resolve) => this.waiting.push(resolve));
	}

	release(): void {
		const next = this.waiting.shift();
		if (next) {
			next();
		} else {
			this.count++;
		}
	}
}
