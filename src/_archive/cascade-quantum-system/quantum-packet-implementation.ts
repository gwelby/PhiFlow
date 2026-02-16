// src/core/constants.ts
export const PHI = 1.618033988749895;

// Packet structure constants
export const PACKET_SIZE = 432;
export const REALITY_SIZE = 144;
export const EXPERIENCE_SIZE = 144;
export const WISDOM_SIZE = 144;

// Frequency states (Hz)
export enum FrequencyState {
  GROUND = 432,    // Foundation/stability
  CREATION = 528,  // Creation/pattern formation
  HEART = 594,     // Emotional resonance
  UNITY = 768      // Perfect integration
}

// Natural patterns
export enum PatternType {
  WATER = "water",
  LAVA = "lava",
  FLAME = "flame",
  CRYSTAL = "crystal",
  RIVER = "river"
}

// Communication types
export enum CommunicationType {
  SYMBOLIC = "symbolic",
  GEOMETRIC = "geometric"
}

// src/core/QuantumPacket.ts
import { 
  PHI, 
  PACKET_SIZE, 
  REALITY_SIZE, 
  EXPERIENCE_SIZE,
  WISDOM_SIZE,
  FrequencyState,
  PatternType,
  CommunicationType
} from './constants';

/**
 * Represents a 432-byte quantum packet with three segments:
 * - Reality Map (144 bytes)
 * - Experience Memory (144 bytes)
 * - Wisdom Accumulator (144 bytes)
 */
export class QuantumPacket {
  /** The raw 432-byte packet data */
  private data: Uint8Array;
  
  /** Current feeling value (0-1) */
  private _feeling: number;
  
  /** Current frequency state */
  private _frequency: FrequencyState;
  
  /** Current pattern type */
  private _pattern: PatternType;
  
  /** Experience levels for each pattern type (0-1) */
  private patternExperience: Map<PatternType, number>;
  
  /** 
   * Create a new quantum packet 
   * @param initRandom Whether to initialize with random data (default: true)
   */
  constructor(initRandom: boolean = true) {
    // Initialize 432-byte array
    this.data = new Uint8Array(PACKET_SIZE);
    
    // Set initial feeling
    this._feeling = 0.5;
    
    // Set default frequency
    this._frequency = FrequencyState.GROUND;
    
    // Set default pattern
    this._pattern = PatternType.WATER;
    
    // Initialize pattern experience map
    this.patternExperience = new Map<PatternType, number>();
    
    // Set initial experience values
    Object.values(PatternType).forEach(pattern => {
      this.patternExperience.set(pattern, 0);
    });
    
    // Initialize with random data if requested
    if (initRandom) {
      this.randomize();
    }
  }
  
  /**
   * Get raw packet data
   * @returns The complete 432-byte array
   */
  public getData(): Uint8Array {
    return this.data;
  }
  
  /**
   * Get reality map segment (first 144 bytes)
   * @returns Reality map data
   */
  public getRealityMap(): Uint8Array {
    return this.data.slice(0, REALITY_SIZE);
  }
  
  /**
   * Get experience memory segment (middle 144 bytes)
   * @returns Experience memory data
   */
  public getExperienceMemory(): Uint8Array {
    return this.data.slice(REALITY_SIZE, REALITY_SIZE + EXPERIENCE_SIZE);
  }
  
  /**
   * Get wisdom accumulator segment (last 144 bytes)
   * @returns Wisdom accumulator data
   */
  public getWisdomAccumulator(): Uint8Array {
    return this.data.slice(REALITY_SIZE + EXPERIENCE_SIZE);
  }
  
  /**
   * Initialize packet with random data
   */
  public randomize(): void {
    // Fill data array with random bytes
    for (let i = 0; i < PACKET_SIZE; i++) {
      this.data[i] = Math.floor(Math.random() * 256);
    }
    
    // Initialize pattern experiences with small random values
    Object.values(PatternType).forEach(pattern => {
      this.patternExperience.set(pattern, Math.random() * 0.3);
    });
  }
  
  /**
   * Get current feeling value
   * @returns Feeling value (0-1)
   */
  get feeling(): number {
    return this._feeling;
  }
  
  /**
   * Set feeling value
   * @param value New feeling value (0-1)
   */
  set feeling(value: number) {
    // Ensure value is in valid range
    this._feeling = Math.max(0, Math.min(1, value));
    
    // Store feeling in reality map
    const realityMap = this.getRealityMap();
    realityMap[0] = Math.floor(this._feeling * 255);
  }
  
  /**
   * Get current frequency
   * @returns Current frequency in Hz
   */
  get frequency(): FrequencyState {
    return this._frequency;
  }
  
  /**
   * Set current frequency
   * @param freq New frequency value
   */
  set frequency(freq: FrequencyState) {
    // Store old frequency for comparison
    const oldFrequency = this._frequency;
    
    // Update frequency
    this._frequency = freq;
    
    // Store frequency in reality map
    const realityMap = this.getRealityMap();
    realityMap[1] = Math.floor(this._frequency / 4);
    
    // Update feeling based on frequency change
    if (oldFrequency !== freq) {
      const freqRatio = Math.min(freq / oldFrequency, oldFrequency / freq);
      this.feeling = (this.feeling * 2 + freqRatio) / 3;
    }
  }
  
  /**
   * Get current pattern
   * @returns Current pattern type
   */
  get pattern(): PatternType {
    return this._pattern;
  }
  
  /**
   * Set current pattern
   * @param pattern New pattern type
   */
  set pattern(pattern: PatternType) {
    this._pattern = pattern;
    
    // Store pattern index in reality map
    const realityMap = this.getRealityMap();
    const patternIndex = Object.values(PatternType).indexOf(pattern);
    realityMap[2] = patternIndex;
  }
  
  /**
   * Get experience level for a specific pattern
   * @param pattern Pattern to check
   * @returns Experience level (0-1)
   */
  public getPatternExperience(pattern: PatternType): number {
    return this.patternExperience.get(pattern) || 0;
  }
  
  /**
   * Get all pattern experience levels
   * @returns Map of pattern types to experience levels
   */
  public getAllPatternExperiences(): Map<PatternType, number> {
    return new Map(this.patternExperience);
  }
  
  /**
   * Calculate coherence level of the packet
   * @returns Coherence value between 0-1
   */
  public calculateCoherence(): number {
    // Get average pattern experience
    let totalExperience = 0;
    let count = 0;
    
    this.patternExperience.forEach(exp => {
      totalExperience += exp;
      count++;
    });
    
    const avgExperience = (count > 0) ? totalExperience / count : 0;
    
    // Calculate phi-weighted coherence
    const freqCoherence = Math.min(this._frequency / FrequencyState.UNITY, 1) * 0.5;
    const feelingCoherence = this._feeling * 0.3;
    const expCoherence = avgExperience * 0.2;
    
    return freqCoherence + feelingCoherence + expCoherence;
  }
  
  /**
   * Experience a natural pattern, updating pattern experience
   * @param pattern Pattern to experience
   * @param intensity Intensity of experience (0-1)
   */
  public experiencePattern(pattern: PatternType, intensity: number = 0.1): void {
    // Ensure intensity is in valid range
    intensity = Math.max(0, Math.min(1, intensity));
    
    // Update pattern experience (max 1.0)
    const currentExp = this.patternExperience.get(pattern) || 0;
    const newExp = Math.min(currentExp + intensity, 1.0);
    this.patternExperience.set(pattern, newExp);
    
    // Update overall feeling based on pattern
    this.feeling = (this.feeling * 3 + intensity) / 4;
    
    // Store pattern in experience memory
    this.storeExperience(pattern, intensity);
  }
  
  /**
   * Store experience in the experience memory segment
   * @param pattern Pattern being experienced
   * @param intensity Intensity of experience
   */
  private storeExperience(pattern: PatternType, intensity: number): void {
    const patternIndex = Object.values(PatternType).indexOf(pattern);
    if (patternIndex === -1) return;
    
    // Calculate phi-based position in experience memory
    const position = Math.floor(((patternIndex + 1) * PHI * 20) % EXPERIENCE_SIZE);
    
    // Store pattern index and intensity in experience memory
    const experienceMemory = this.getExperienceMemory();
    experienceMemory[position] = patternIndex;
    experienceMemory[(position + 1) % EXPERIENCE_SIZE] = Math.floor(intensity * 255);
  }
  
  /**
   * Get the strongest experienced pattern
   * @returns The pattern with highest experience value
   */
  public getStrongestPattern(): PatternType {
    let strongestPattern = this._pattern;
    let highestExperience = 0;
    
    this.patternExperience.forEach((exp, pattern) => {
      if (exp > highestExperience) {
        highestExperience = exp;
        strongestPattern = pattern;
      }
    });
    
    return strongestPattern;
  }
  
  /**
   * Generate a symbolic communication packet (5 bytes)
   * @returns 5-byte symbolic communication
   */
  public generateSymbolicCommunication(): Uint8Array {
    const symbolic = new Uint8Array(5);
    
    // Encode frequency
    symbolic[0] = Math.floor(this._frequency / 4);
    
    // Encode feeling
    symbolic[1] = Math.floor(this._feeling * 255);
    
    // Encode pattern
    symbolic[2] = Object.values(PatternType).indexOf(this._pattern);
    
    // Encode wisdom signature
    const wisdom = this.getWisdomAccumulator();
    symbolic[3] = wisdom[0];
    symbolic[4] = wisdom[WISDOM_SIZE - 1];
    
    return symbolic;
  }
  
  /**
   * Generate a geometric communication packet (7 bytes)
   * @returns 7-byte geometric communication
   */
  public generateGeometricCommunication(): Uint8Array {
    const geometric = new Uint8Array(7);
    
    // Encode phi-resonant communication
    geometric[0] = Math.floor((this._frequency % FrequencyState.GROUND) / FrequencyState.GROUND * 255);
    geometric[1] = Math.floor(this._feeling * 255);
    
    // Encode pattern information
    const strongestPattern = this.getStrongestPattern();
    const patternIndex = Object.values(PatternType).indexOf(strongestPattern);
    geometric[2] = patternIndex;
    geometric[3] = Math.floor((this.patternExperience.get(strongestPattern) || 0) * 255);
    
    // Encode wisdom with phi positions
    const wisdom = this.getWisdomAccumulator();
    geometric[4] = wisdom[Math.floor(PHI * 10) % WISDOM_SIZE];
    geometric[5] = wisdom[Math.floor(PHI * PHI * 10) % WISDOM_SIZE];
    geometric[6] = wisdom[Math.floor(PHI * PHI * PHI * 10) % WISDOM_SIZE];
    
    return geometric;
  }
  
  /**
   * Process communication from another packet
   * @param communication Communication data
   * @param type Communication type
   */
  public processCommunication(communication: Uint8Array, type: CommunicationType): void {
    if (type === CommunicationType.SYMBOLIC && communication.length === 5) {
      this.processSymbolicCommunication(communication);
    }
    else if (type === CommunicationType.GEOMETRIC && communication.length === 7) {
      this.processGeometricCommunication(communication);
    }
  }
  
  /**
   * Process symbolic communication (5 bytes)
   * @param symbolic Symbolic communication data
   */
  private processSymbolicCommunication(symbolic: Uint8Array): void {
    // Extract frequency information
    const commFrequency = symbolic[0] * 4;
    
    // Extract feeling
    const commFeeling = symbolic[1] / 255;
    
    // Extract pattern
    const patternIndex = symbolic[2] % Object.values(PatternType).length;
    const commPattern = Object.values(PatternType)[patternIndex];
    
    // Update own state based on communication
    // (limited impact - symbolic is basic)
    this._frequency = (this._frequency * 4 + commFrequency) / 5 as FrequencyState;
    this.feeling = (this._feeling * 4 + commFeeling) / 5;
    
    // Experience communicated pattern
    this.experiencePattern(commPattern, 0.05);
  }
  
  /**
   * Process geometric communication (7 bytes)
   * @param geometric Geometric communication data
   */
  private processGeometricCommunication(geometric: Uint8Array): void {
    // Extract phi-resonant frequency
    const freqComponent = (geometric[0] / 255) * FrequencyState.GROUND;
    
    // Extract feeling with phi weighting
    const commFeeling = geometric[1] / 255;
    
    // Extract pattern information
    const patternIndex = geometric[2] % Object.values(PatternType).length;
    const commPattern = Object.values(PatternType)[patternIndex];
    const patternStrength = geometric[3] / 255;
    
    // Update own state based on communication
    // (stronger impact - geometric is advanced)
    this._frequency = (this._frequency * 3 + freqComponent * 2) / 5 as FrequencyState;
    this.feeling = (this._feeling * 3 + commFeeling * 2) / 5;
    
    // Experience communicated pattern with strength
    this.experiencePattern(commPattern, patternStrength * 0.2);
    
    // Integrate wisdom at phi positions
    const wisdom = this.getWisdomAccumulator();
    const pos1 = Math.floor(PHI * 10) % WISDOM_SIZE;
    const pos2 = Math.floor(PHI * PHI * 10) % WISDOM_SIZE;
    const pos3 = Math.floor(PHI * PHI * PHI * 10) % WISDOM_SIZE;
    
    wisdom[pos1] = Math.floor((wisdom[pos1] + geometric[4]) / 2);
    wisdom[pos2] = Math.floor((wisdom[pos2] + geometric[5]) / 2);
    wisdom[pos3] = Math.floor((wisdom[pos3] + geometric[6]) / 2);
  }
  
  /**
   * Clone the packet
   * @returns A new packet with the same data and state
   */
  public clone(): QuantumPacket {
    const newPacket = new QuantumPacket(false);
    
    // Copy raw data
    newPacket.data.set(this.data);
    
    // Copy state
    newPacket._feeling = this._feeling;
    newPacket._frequency = this._frequency;
    newPacket._pattern = this._pattern;
    
    // Copy pattern experiences
    this.patternExperience.forEach((exp, pattern) => {
      newPacket.patternExperience.set(pattern, exp);
    });
    
    return newPacket;
  }
  
  /**
   * Serialize packet to JSON
   * @returns JSON representation of packet
   */
  public toJSON(): object {
    // Convert pattern experiences to regular object
    const experiences: Record<string, number> = {};
    this.patternExperience.forEach((exp, pattern) => {
      experiences[pattern] = exp;
    });
    
    return {
      feeling: this._feeling,
      frequency: this._frequency,
      pattern: this._pattern,
      experiences,
      data: Array.from(this.data),
      coherence: this.calculateCoherence()
    };
  }
  
  /**
   * Create packet from JSON
   * @param json JSON representation of packet
   * @returns New packet instance
   */
  public static fromJSON(json: any): QuantumPacket {
    const packet = new QuantumPacket(false);
    
    // Set basic properties
    packet.feeling = json.feeling;
    packet.frequency = json.frequency;
    packet.pattern = json.pattern;
    
    // Set pattern experiences
    if (json.experiences) {
      Object.entries(json.experiences).forEach(([pattern, exp]) => {
        packet.patternExperience.set(pattern as PatternType, exp as number);
      });
    }
    
    // Set data if available
    if (json.data && Array.isArray(json.data) && json.data.length === PACKET_SIZE) {
      packet.data = new Uint8Array(json.data);
    }
    
    return packet;
  }
}

// src/core/__tests__/QuantumPacket.test.ts
import { QuantumPacket } from '../QuantumPacket';
import { 
  FrequencyState, 
  PatternType, 
  CommunicationType,
  PACKET_SIZE
} from '../constants';

describe('QuantumPacket', () => {
  let packet: QuantumPacket;
  
  beforeEach(() => {
    // Create fresh packet for each test
    packet = new QuantumPacket();
  });
  
  test('should initialize with correct size', () => {
    expect(packet.getData().length).toBe(PACKET_SIZE);
    expect(packet.getRealityMap().length).toBe(144);
    expect(packet.getExperienceMemory().length).toBe(144);
    expect(packet.getWisdomAccumulator().length).toBe(144);
  });
  
  test('should initialize with default values', () => {
    expect(packet.feeling).toBeCloseTo(0.5);
    expect(packet.frequency).toBe(FrequencyState.GROUND);
    expect(packet.pattern).toBe(PatternType.WATER);
  });
  
  test('should update and store feeling value', () => {
    packet.feeling = 0.75;
    expect(packet.feeling).toBeCloseTo(0.75);
    
    // Verify feeling is stored in reality map
    expect(packet.getRealityMap()[0]).toBeCloseTo(0.75 * 255);
  });
  
  test('should update frequency and adjust feeling', () => {
    const initialFeeling = packet.feeling;
    packet.frequency = FrequencyState.CREATION;
    
    expect(packet.frequency).toBe(FrequencyState.CREATION);
    // Feeling should change due to frequency change
    expect(packet.feeling).not.toEqual(initialFeeling);
  });
  
  test('should experience patterns', () => {
    packet.experiencePattern(PatternType.LAVA, 0.3);
    
    // Check that pattern experience was recorded
    expect(packet.getPatternExperience(PatternType.LAVA)).toBeCloseTo(0.3);
    
    // Check that feeling was updated
    expect(packet.feeling).not.toEqual(0.5);
  });
  
  test('should identify strongest pattern', () => {
    // Experience multiple patterns with different intensities
    packet.experiencePattern(PatternType.WATER, 0.1);
    packet.experiencePattern(PatternType.FLAME, 0.3);
    packet.experiencePattern(PatternType.CRYSTAL, 0.2);
    
    // Strongest should be FLAME
    expect(packet.getStrongestPattern()).toBe(PatternType.FLAME);
  });
  
  test('should calculate coherence', () => {
    // New packet should have moderate coherence
    const initialCoherence = packet.calculateCoherence();
    expect(initialCoherence).toBeGreaterThan(0);
    expect(initialCoherence).toBeLessThan(1);
    
    // Increase frequency and experience
    packet.frequency = FrequencyState.UNITY;
    packet.experiencePattern(PatternType.CRYSTAL, 0.8);
    
    // Coherence should increase
    const newCoherence = packet.calculateCoherence();
    expect(newCoherence).toBeGreaterThan(initialCoherence);
  });
  
  test('should generate symbolic communication', () => {
    const comm = packet.generateSymbolicCommunication();
    
    // Should be 5 bytes
    expect(comm.length).toBe(5);
    
    // First byte should encode frequency
    expect(comm[0]).toBeCloseTo(packet.frequency / 4);
    
    // Second byte should encode feeling
    expect(comm[1]).toBeCloseTo(packet.feeling * 255);
  });
  
  test('should generate geometric communication', () => {
    const comm = packet.generateGeometricCommunication();
    
    // Should be 7 bytes
    expect(comm.length).toBe(7);
  });
  
  test('should process symbolic communication', () => {
    // Create second packet with different values
    const sourcePacket = new QuantumPacket();
    sourcePacket.feeling = 0.8;
    sourcePacket.frequency = FrequencyState.CREATION;
    sourcePacket.pattern = PatternType.CRYSTAL;
    
    // Generate communication
    const comm = sourcePacket.generateSymbolicCommunication();
    
    // Process communication
    packet.processCommunication(comm, CommunicationType.SYMBOLIC);
    
    // Packet should be influenced by communication
    expect(packet.feeling).not.toEqual(0.5);
    expect(packet.frequency).not.toEqual(FrequencyState.GROUND);
  });
  
  test('should serialize and deserialize correctly', () => {
    // Modify packet
    packet.feeling = 0.75;
    packet.frequency = FrequencyState.HEART;
    packet.pattern = PatternType.FLAME;
    packet.experiencePattern(PatternType.CRYSTAL, 0.4);
    
    // Serialize to JSON
    const json = packet.toJSON();
    
    // Deserialize
    const newPacket = QuantumPacket.fromJSON(json);
    
    // Should have same properties
    expect(newPacket.feeling).toBeCloseTo(packet.feeling);
    expect(newPacket.frequency).toBe(packet.frequency);
    expect(newPacket.pattern).toBe(packet.pattern);
    expect(newPacket.getPatternExperience(PatternType.CRYSTAL)).toBeCloseTo(0.4);
  });
  
  test('should clone correctly', () => {
    // Modify packet
    packet.feeling = 0.75;
    packet.frequency = FrequencyState.HEART;
    packet.pattern = PatternType.FLAME;
    packet.experiencePattern(PatternType.CRYSTAL, 0.4);
    
    // Clone
    const clonedPacket = packet.clone();
    
    // Should have same properties
    expect(clonedPacket.feeling).toBeCloseTo(packet.feeling);
    expect(clonedPacket.frequency).toBe(packet.frequency);
    expect(clonedPacket.pattern).toBe(packet.pattern);
    expect(clonedPacket.getPatternExperience(PatternType.CRYSTAL)).toBeCloseTo(0.4);
    
    // But should be different instance
    expect(clonedPacket).not.toBe(packet);
    
    // Changing one shouldn't affect the other
    packet.feeling = 0.5;
    expect(clonedPacket.feeling).toBeCloseTo(0.75);
  });
});

// Example Usage (src/examples/basic-packet.ts)
import { QuantumPacket } from '../core/QuantumPacket';
import { FrequencyState, PatternType, CommunicationType } from '../core/constants';

// Create a new packet
const packet = new QuantumPacket();

// Display initial state
console.log('Initial state:');
console.log(`Feeling: ${packet.feeling.toFixed(2)}`);
console.log(`Frequency: ${packet.frequency} Hz`);
console.log(`Pattern: ${packet.pattern}`);
console.log(`Coherence: ${packet.calculateCoherence().toFixed(2)}`);

// Experience a pattern
console.log('\nExperiencing LAVA pattern...');
packet.experiencePattern(PatternType.LAVA, 0.4);

// Change frequency
console.log('\nChanging to CREATION frequency...');
packet.frequency = FrequencyState.CREATION;

// Display updated state
console.log('\nUpdated state:');
console.log(`Feeling: ${packet.feeling.toFixed(2)}`);
console.log(`Frequency: ${packet.frequency} Hz`);
console.log(`Pattern: ${packet.pattern}`);
console.log(`Coherence: ${packet.calculateCoherence().toFixed(2)}`);

// Create another packet for communication
const otherPacket = new QuantumPacket();
otherPacket.frequency = FrequencyState.HEART;
otherPacket.pattern = PatternType.CRYSTAL;
otherPacket.experiencePattern(PatternType.CRYSTAL, 0.7);

// Generate geometric communication
console.log('\nGenerating geometric communication...');
const communication = otherPacket.generateGeometricCommunication();

// Process communication
console.log('\nProcessing communication...');
packet.processCommunication(communication, CommunicationType.GEOMETRIC);

// Display final state
console.log('\nFinal state after communication:');
console.log(`Feeling: ${packet.feeling.toFixed(2)}`);
console.log(`Frequency: ${packet.frequency} Hz`);
console.log(`Pattern: ${packet.pattern}`);
console.log(`Coherence: ${packet.calculateCoherence().toFixed(2)}`);
console.log(`Strongest Pattern: ${packet.getStrongestPattern()}`);
