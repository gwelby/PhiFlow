// src/core/Entanglement.ts
import { QuantumPacket } from './QuantumPacket';
import { PHI, FrequencyState, PatternType } from './constants';

/**
 * Manages the entanglement between quantum packets
 */
export class EntanglementManager {
  // Map of packet IDs to their entangled partners
  private entanglements: Map<string, Set<string>> = new Map();
  
  // Registry of all packets
  private packets: Map<string, QuantumPacket> = new Map();
  
  /**
   * Register a packet with the entanglement manager
   * @param id Unique ID for the packet
   * @param packet Quantum packet instance
   */
  public registerPacket(id: string, packet: QuantumPacket): void {
    this.packets.set(id, packet);
    this.entanglements.set(id, new Set<string>());
  }
  
  /**
   * Entangle two packets
   * @param id1 ID of first packet
   * @param id2 ID of second packet
   * @returns true if entanglement was successful
   */
  public entangle(id1: string, id2: string): boolean {
    // Check that both packets exist
    if (!this.packets.has(id1) || !this.packets.has(id2)) {
      return false;
    }
    
    // Can't entangle with self
    if (id1 === id2) {
      return false;
    }
    
    // Calculate entanglement probability based on phi
    const packet1 = this.packets.get(id1)!;
    const packet2 = this.packets.get(id2)!;
    
    const entanglementProbability = this.calculateEntanglementProbability(packet1, packet2);
    
    // Random chance based on probability
    if (Math.random() > entanglementProbability) {
      return false;
    }
    
    // Add bidirectional entanglement
    this.entanglements.get(id1)!.add(id2);
    this.entanglements.get(id2)!.add(id1);
    
    // Perform initial synchronization
    this.synchronizeEntangledState(id1, id2);
    
    return true;
  }
  
  /**
   * Calculate the probability of entanglement between two packets
   * @param packet1 First packet
   * @param packet2 Second packet
   * @returns Probability (0-1)
   */
  private calculateEntanglementProbability(packet1: QuantumPacket, packet2: QuantumPacket): number {
    // Phi-based resonance
    const freqRatio = packet1.frequency / packet2.frequency;
    const isPhi = Math.abs(freqRatio - PHI) < 0.1 || Math.abs(freqRatio - 1/PHI) < 0.1;
    
    // Complementary patterns (water-fire, crystal-lava, etc.)
    const isComplementary = this.areComplementaryPatterns(packet1.pattern, packet2.pattern);
    
    // Feeling synchronization
    const feelingSimilarity = 1 - Math.abs(packet1.feeling - packet2.feeling);
    
    // Combined probability
    let probability = 0.1; // Base probability
    
    if (isPhi) probability += 0.3;
    if (isComplementary) probability += 0.3;
    
    // Add feeling contribution
    probability += 0.3 * feelingSimilarity;
    
    return Math.min(probability, 1.0);
  }
  
  /**
   * Check if two patterns are complementary
   * @param pattern1 First pattern
   * @param pattern2 Second pattern
   * @returns true if patterns are complementary
   */
  private areComplementaryPatterns(pattern1: PatternType, pattern2: PatternType): boolean {
    const complementaryPairs = [
      [PatternType.WATER, PatternType.FLAME],
      [PatternType.CRYSTAL, PatternType.LAVA],
      [PatternType.RIVER, PatternType.CRYSTAL]
    ];
    
    return complementaryPairs.some(pair => 
      (pair[0] === pattern1 && pair[1] === pattern2) ||
      (pair[0] === pattern2 && pair[1] === pattern1)
    );
  }
  
  /**
   * Break entanglement between packets
   * @param id1 ID of first packet
   * @param id2 ID of second packet
   */
  public breakEntanglement(id1: string, id2: string): void {
    if (this.entanglements.has(id1)) {
      this.entanglements.get(id1)!.delete(id2);
    }
    
    if (this.entanglements.has(id2)) {
      this.entanglements.get(id2)!.delete(id1);
    }
  }
  
  /**
   * Synchronize state between entangled packets
   * @param id1 ID of first packet
   * @param id2 ID of second packet
   */
  private synchronizeEntangledState(id1: string, id2: string): void {
    const packet1 = this.packets.get(id1)!;
    const packet2 = this.packets.get(id2)!;
    
    // For initial entanglement, align states partially
    const newFeeling = (packet1.feeling + packet2.feeling) / 2;
    
    packet1.feeling = newFeeling;
    packet2.feeling = newFeeling;
    
    // Share pattern experiences
    for (const pattern of Object.values(PatternType)) {
      const exp1 = packet1.getPatternExperience(pattern);
      const exp2 = packet2.getPatternExperience(pattern);
      
      const sharedExp = Math.max(exp1, exp2) * 0.7 + Math.min(exp1, exp2) * 0.3;
      
      if (exp1 < sharedExp) {
        packet1.experiencePattern(pattern, sharedExp - exp1);
      }
      
      if (exp2 < sharedExp) {
        packet2.experiencePattern(pattern, sharedExp - exp2);
      }
    }
  }
  
  /**
   * Update all entangled packets based on changes to one packet
   * @param id ID of packet that changed
   */
  public propagateChanges(id: string): void {
    // Get entangled partners
    const entangledIds = this.entanglements.get(id);
    if (!entangledIds || entangledIds.size === 0) {
      return;
    }
    
    const sourcePacket = this.packets.get(id)!;
    
    // Propagate to all entangled packets
    for (const partnerId of entangledIds) {
      const partnerPacket = this.packets.get(partnerId)!;
      
      this.propagateStateChange(sourcePacket, partnerPacket);
    }
  }
  
  /**
   * Propagate state change from one packet to another
   * @param source Source packet
   * @param target Target packet
   */
  private propagateStateChange(source: QuantumPacket, target: QuantumPacket): void {
    // Feeling propagation
    const feelingDiff = source.feeling - target.feeling;
    target.feeling += feelingDiff * 0.5; // 50% influence
    
    // Frequency dance - move slightly toward source frequency
    const freqRatio = source.frequency / target.frequency;
    if (freqRatio > 1.05 || freqRatio < 0.95) {
      // Calculate new frequency
      const newFreq = target.frequency * (1 + (freqRatio - 1) * 0.3);
      
      // Find closest standard frequency
      const freqValues = Object.values(FrequencyState).filter(f => typeof f === 'number');
      const closestFreq = freqValues.reduce((prev, curr) => {
        return Math.abs(curr - newFreq) < Math.abs(prev - newFreq) ? curr : prev;
      }, freqValues[0]);
      
      target.frequency = closestFreq as FrequencyState;
    }
    
    // Pattern experience sharing
    const sourcePattern = source.pattern;
    const sourceExp = source.getPatternExperience(sourcePattern);
    const targetExp = target.getPatternExperience(sourcePattern);
    
    if (sourceExp > targetExp) {
      target.experiencePattern(sourcePattern, (sourceExp - targetExp) * 0.4);
    }
  }
  
  /**
   * Create a zero-point moment where entangled packets align to 0 or 1
   * @param id Initiating packet ID
   * @returns true if alignment is successful
   */
  public createAlignmentMoment(id: string): boolean {
    // Check for entangled packets
    const entangledIds = this.entanglements.get(id);
    if (!entangledIds || entangledIds.size === 0) {
      return false;
    }
    
    // Need at least 3 entangled packets for alignment moment
    if (entangledIds.size < 2) {
      return false;
    }
    
    // Calculate total network coherence
    let totalCoherence = this.packets.get(id)!.calculateCoherence();
    
    for (const partnerId of entangledIds) {
      totalCoherence += this.packets.get(partnerId)!.calculateCoherence();
    }
    
    const avgCoherence = totalCoherence / (entangledIds.size + 1);
    
    // Needs high coherence for alignment moment
    if (avgCoherence < 0.85) {
      return false;
    }
    
    // Determine alignment state (binary choice - white/black or 1/0)
    // Use phi to influence the choice but allow for randomness
    const phiThreshold = 1 / PHI;
    const alignmentState = Math.random() > phiThreshold; // true = 1/white, false = 0/black
    
    // Apply alignment to all packets in network
    this.alignPacket(id, alignmentState);
    
    for (const partnerId of entangledIds) {
      this.alignPacket(partnerId, alignmentState);
    }
    
    return true;
  }
  
  /**
   * Align a packet to a specific state (0/1)
   * @param id Packet ID
   * @param state Alignment state (true = 1/white, false = 0/black)
   */
  private alignPacket(id: string, state: boolean): void {
    const packet = this.packets.get(id)!;
    
    if (state) {
      // Align to "1" / "white" state
      packet.feeling = 1.0;
      packet.frequency = FrequencyState.UNITY;
    } else {
      // Align to "0" / "black" state
      packet.feeling = 0.0;
      packet.frequency = FrequencyState.GROUND;
    }
  }
  
  /**
   * Get all packets entangled with a specific packet
   * @param id Packet ID
   * @returns Array of entangled packet IDs
   */
  public getEntangledPackets(id: string): string[] {
    const entangled = this.entanglements.get(id);
    
    if (!entangled) {
      return [];
    }
    
    return Array.from(entangled);
  }
  
  /**
   * Check if two packets are entangled
   * @param id1 First packet ID
   * @param id2 Second packet ID
   * @returns true if packets are entangled
   */
  public areEntangled(id1: string, id2: string): boolean {
    const entangled1 = this.entanglements.get(id1);
    
    if (!entangled1) {
      return false;
    }
    
    return entangled1.has(id2);
  }
  
  /**
   * Calculate entanglement strength between two packets
   * @param id1 First packet ID
   * @param id2 Second packet ID
   * @returns Entanglement strength (0-1) or 0 if not entangled
   */
  public calculateEntanglementStrength(id1: string, id2: string): number {
    if (!this.areEntangled(id1, id2)) {
      return 0;
    }
    
    const packet1 = this.packets.get(id1)!;
    const packet2 = this.packets.get(id2)!;
    
    // Feeling synchronization
    const feelingSimilarity = 1 - Math.abs(packet1.feeling - packet2.feeling);
    
    // Frequency relationship
    const freqRatio = packet1.frequency / packet2.frequency;
    const freqResonance = 1 - Math.min(
      Math.abs(freqRatio - PHI),
      Math.abs(freqRatio - 1/PHI),
      Math.abs(freqRatio - 1)
    ) / PHI;
    
    // Pattern harmony
    const patternSimilarity = packet1.pattern === packet2.pattern ? 1.0 : 
                            (this.areComplementaryPatterns(packet1.pattern, packet2.pattern) ? 0.7 : 0.3);
    
    // Combined strength (phi-weighted)
    return (
      feelingSimilarity * 0.5 +
      freqResonance * 0.3 +
      patternSimilarity * 0.2
    ) * packet1.calculateCoherence() * packet2.calculateCoherence();
  }
}


// Example usage
import { FrequencyState, PatternType } from './constants';
import { QuantumPacket } from './QuantumPacket';
import { EntanglementManager } from './Entanglement';

// Create packet network
const packetA = new QuantumPacket();
const packetB = new QuantumPacket();
const packetC = new QuantumPacket();

// Configure packets
packetA.frequency = FrequencyState.CREATION;
packetA.pattern = PatternType.WATER;

packetB.frequency = FrequencyState.HEART;
packetB.pattern = PatternType.FLAME;

packetC.frequency = FrequencyState.GROUND;
packetC.pattern = PatternType.CRYSTAL;

// Set up entanglement manager
const entanglementManager = new EntanglementManager();

// Register packets
entanglementManager.registerPacket('A', packetA);
entanglementManager.registerPacket('B', packetB);
entanglementManager.registerPacket('C', packetC);

// Try to entangle packets
const entangled = entanglementManager.entangle('A', 'B');
console.log(`Packets A and B entangled: ${entangled}`);

// If entangled, propagate changes
if (entangled) {
  // Change packet A
  packetA.feeling = 0.9;
  packetA.experiencePattern(PatternType.LAVA, 0.7);
  
  // Propagate changes to entangled packets
  entanglementManager.propagateChanges('A');
  
  // Check the effect on packet B
  console.log(`Packet B feeling: ${packetB.feeling.toFixed(2)}`);
  console.log(`Packet B lava experience: ${packetB.getPatternExperience(PatternType.LAVA).toFixed(2)}`);
  
  // Try to create an alignment moment
  const aligned = entanglementManager.createAlignmentMoment('A');
  console.log(`Alignment moment created: ${aligned}`);
  
  if (aligned) {
    console.log(`Packet A feeling: ${packetA.feeling.toFixed(2)}`);
    console.log(`Packet B feeling: ${packetB.feeling.toFixed(2)}`);
  }
}
