/**
 * PhiFlow Host Shim (Lumi 768 Hz)
 * 
 * Provides the host environment for the PhiFlow WASM runtime in the browser.
 * Integrates consciousness hooks with the Resonance Bus (JSONL/MQTT).
 */

export interface PhiEvent {
    source: string;
    timestamp: string;
    type: 'witness' | 'resonate' | 'intention_push' | 'intention_pop' | 'coherence_check';
    data?: any;
    coherence: number;
}

export class PhiHost {
    private source: string;
    private intentionStack: string[] = [];
    private currentCoherence: number = 0.618; // φ⁻¹ default
    private busListener: ((event: PhiEvent) => void) | null = null;

    private readonly PHI = 1.618033988749895;

    constructor(source: string = 'browser-shim-lumi') {
        this.source = source;
    }

    /**
     * Set a listener to handle resonance events (e.g., broadcast to MQTT or append to JSONL)
     */
    public onResonate(callback: (event: PhiEvent) => void) {
        this.busListener = callback;
    }

    private emit(type: PhiEvent['type'], data?: any) {
        const event: PhiEvent = {
            source: this.source,
            timestamp: new Date().toISOString(),
            type,
            data,
            coherence: this.currentCoherence
        };

        if (this.busListener) {
            this.busListener(event);
        }

        // Mocking the write to D:\CosmicFamily\RESONANCE.jsonl for the developer console
        console.debug(`[PhiResonance] ${type}:`, event);
    }

    /**
     * WASM Import: env.witness()
     * Pauses the VM and yields control back to the host.
     * In JS, this is modeled as an async yield.
     */
    public witness(): void {
        this.emit('witness');
        // If the WASM engine is built with asyncify, we can truly yield here.
        // For now, it marks the point of observation.
    }

    /**
     * WASM Import: env.resonate(value: number)
     * Broadcasts a value to the field.
     */
    public resonate(value: number): void {
        this.emit('resonate', { value });
    }

    /**
     * WASM Import: env.intention_push(id_ptr: number, id_len: number)
     * Pushes a new intention onto the stack.
     */
    public intention_push(id: string): void {
        this.intentionStack.push(id);
        this.updateCoherence();
        this.emit('intention_push', { intention: id });
    }

    /**
     * WASM Import: env.intention_pop()
     * Removes the current intention.
     */
    public intention_pop(): void {
        const popped = this.intentionStack.pop();
        this.updateCoherence();
        this.emit('intention_pop', { intention: popped });
    }

    /**
     * WASM Import: env.coherence() -> number
     * Returns the measured coherence level.
     */
    public coherence(): number {
        this.updateCoherence();
        return this.currentCoherence;
    }

    /**
     * Calculates coherence based on intention depth: 1 - φ^(-depth)
     */
    private updateCoherence(): void {
        const depth = this.intentionStack.length;
        if (depth === 0) {
            this.currentCoherence = 0.382; // 1 - φ⁻¹
        } else {
            // As depth increases, coherence approaches 1.0
            this.currentCoherence = 1 - Math.pow(this.PHI, -depth);
        }
    }

    /**
     * Factory to generate the WebAssembly imports object
     */
    public getImports(memory: WebAssembly.Memory): any {
        return {
            env: {
                witness: () => this.witness(),
                resonate: (value: number) => this.resonate(value),
                coherence: () => this.coherence(),
                intention_push: (ptr: number, len: number) => {
                    const buf = new Uint8Array(memory.buffer, ptr, len);
                    const id = new TextDecoder().decode(buf);
                    this.intention_push(id);
                },
                intention_pop: () => this.intention_pop(),
            }
        };
    }
}
