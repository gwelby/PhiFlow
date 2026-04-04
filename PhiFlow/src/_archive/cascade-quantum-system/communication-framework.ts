// src/communication/CommunicationHub.ts
import { 
  QuantumPacket,
  FrequencyState,
  PatternType, 
  CommunicationType,
  PHI 
} from '../core';

/**
 * Represents a communication event between two quantum packets
 */
export interface CommunicationEvent {
  /** Source packet ID */
  sourceId: string;
  
  /** Target packet ID */
  targetId: string;
  
  /** Communication type */
  type: CommunicationType;
  
  /** Communication data */
  data: Uint8Array;
  
  /** Timestamp of communication */
  timestamp: number;
  
  /** Communication strength (0-1) */
  strength: number;
  
  /** Whether communication was successful */
  successful: boolean;
}

/**
 * Options for establishing communication
 */
export interface CommunicationOptions {
  /** Communication type (default: SYMBOLIC) */
  type?: CommunicationType;
  
  /** Minimum coherence level required (default: 0.3) */
  minCoherence?: number;
  
  /** Required frequency ratio (default: any) */
  frequencyRatio?: number;
  
  /** Required pattern match (default: any) */
  requirePatternMatch?: boolean;
}

/**
 * Options for creating a connection
 */
export interface ConnectionOptions {
  /** Connection strength (0-1, default: 0.5) */
  strength?: number;
  
  /** Duration in milliseconds (default: permanent) */
  duration?: number;
  
  /** Whether connection is bidirectional (default: true) */
  bidirectional?: boolean;
}

/**
 * Represents a connection between two quantum packets
 */
export interface PacketConnection {
  /** Source packet ID */
  sourceId: string;
  
  /** Target packet ID */
  targetId: string;
  
  /** Connection strength (0-1) */
  strength: number;
  
  /** When connection was established */
  established: number;
  
  /** Duration in milliseconds (0 = permanent) */
  duration: number;
  
  /** Whether connection is bidirectional */
  bidirectional: boolean;
  
  /** Communication events that have occurred on this connection */
  events: CommunicationEvent[];
  
  /** Whether connection is currently active */
  active: boolean;
}

/**
 * Configuration for the communication hub
 */
export interface CommunicationHubConfig {
  /** How often to process communications (milliseconds) */
  processingInterval?: number;
  
  /** Whether to automatically establish connections */
  autoConnections?: boolean;
  
  /** Minimum coherence for auto-connections */
  autoConnectionCoherence?: number;
  
  /** Maximum connections per packet */
  maxConnectionsPerPacket?: number;
  
  /** Default communication type */
  defaultCommunicationType?: CommunicationType;
  
  /** Whether to use phi-optimized timing */
  usePhiTiming?: boolean;
}

/**
 * Listener for communication events
 */
export type CommunicationListener = (event: CommunicationEvent) => void;

/**
 * Manages communication between quantum packets
 */
export class CommunicationHub {
  /** Map of packet IDs to packet instances */
  private packets: Map<string, QuantumPacket> = new Map();
  
  /** Active connections between packets */
  private connections: PacketConnection[] = [];
  
  /** Communication event listeners */
  private listeners: CommunicationListener[] = [];
  
  /** Processing interval ID */
  private processingInterval: number | null = null;
  
  /** Configuration */
  private config: CommunicationHubConfig;
  
  /** Communication statistics */
  private stats = {
    totalCommunications: 0,
    successfulCommunications: 0,
    failedCommunications: 0,
    symbolicCommunications: 0,
    geometricCommunications: 0
  };
  
  /**
   * Create a new communication hub
   * @param config Configuration options
   */
  constructor(config: CommunicationHubConfig = {}) {
    this.config = {
      processingInterval: 1000,       // Process every second
      autoConnections: true,          // Auto-establish connections
      autoConnectionCoherence: 0.5,   // Minimum coherence for auto-connections
      maxConnectionsPerPacket: 5,     // Max connections per packet
      defaultCommunicationType: CommunicationType.SYMBOLIC, // Default type
      usePhiTiming: true,             // Use phi-based timing
      ...config
    };
  }
  
  /**
   * Start the communication hub
   */
  public start(): void {
    if (this.processingInterval !== null) {
      this.stop();
    }
    
    // Use phi-optimized timing if enabled
    const interval = this.config.usePhiTiming 
      ? this.config.processingInterval! / PHI 
      : this.config.processingInterval;
      
    this.processingInterval = window.setInterval(() => {
      this.processAllCommunications();
    }, interval);
    
    console.log(`Communication hub started (interval: ${interval}ms)`);
  }
  
  /**
   * Stop the communication hub
   */
  public stop(): void {
    if (this.processingInterval !== null) {
      window.clearInterval(this.processingInterval);
      this.processingInterval = null;
      console.log('Communication hub stopped');
    }
  }
  
  /**
   * Register a packet with the hub
   * @param id Unique packet ID
   * @param packet Quantum packet instance
   */
  public registerPacket(id: string, packet: QuantumPacket): void {
    if (this.packets.has(id)) {
      console.warn(`Packet with ID ${id} already registered`);
      return;
    }
    
    this.packets.set(id, packet);
    
    // If auto-connections enabled, try to establish with existing packets
    if (this.config.autoConnections) {
      this.attemptAutoConnections(id);
    }
    
    console.log(`Packet ${id} registered with communication hub`);
  }
  
  /**
   * Unregister a packet from the hub
   * @param id Packet ID
   */
  public unregisterPacket(id: string): void {
    if (!this.packets.has(id)) {
      console.warn(`Packet with ID ${id} not registered`);
      return;
    }
    
    // Remove connections involving this packet
    this.connections = this.connections.filter(
      conn => conn.sourceId !== id && conn.targetId !== id
    );
    
    this.packets.delete(id);
    console.log(`Packet ${id} unregistered from communication hub`);
  }
  
  /**
   * Get all registered packets
   * @returns Map of packet IDs to packets
   */
  public getPackets(): Map<string, QuantumPacket> {
    return new Map(this.packets);
  }
  
  /**
   * Get a specific packet
   * @param id Packet ID
   * @returns Packet instance or undefined
   */
  public getPacket(id: string): QuantumPacket | undefined {
    return this.packets.get(id);
  }
  
  /**
   * Establish a connection between two packets
   * @param sourceId Source packet ID
   * @param targetId Target packet ID
   * @param options Connection options
   * @returns Whether connection was established
   */
  public establishConnection(
    sourceId: string, 
    targetId: string, 
    options: ConnectionOptions = {}
  ): boolean {
    // Validate packets exist
    const sourcePacket = this.packets.get(sourceId);
    const targetPacket = this.packets.get(targetId);
    
    if (!sourcePacket || !targetPacket) {
      console.warn('Cannot establish connection: one or both packets not found');
      return false;
    }
    
    // Check if connection already exists
    const existingConnection = this.connections.find(
      conn => conn.sourceId === sourceId && conn.targetId === targetId
    );
    
    if (existingConnection) {
      console.warn('Connection already exists between packets');
      return false;
    }
    
    // Check max connections limit
    const sourceConnections = this.getPacketConnections(sourceId);
    if (sourceConnections.length >= this.config.maxConnectionsPerPacket!) {
      console.warn(`Source packet ${sourceId} has reached max connections`);
      return false;
    }
    
    const targetConnections = this.getPacketConnections(targetId);
    if (targetConnections.length >= this.config.maxConnectionsPerPacket!) {
      console.warn(`Target packet ${targetId} has reached max connections`);
      return false;
    }
    
    // Default options
    const finalOptions: Required<ConnectionOptions> = {
      strength: 0.5,
      duration: 0,  // Permanent
      bidirectional: true,
      ...options
    };
    
    // Create connection
    const connection: PacketConnection = {
      sourceId,
      targetId,
      strength: finalOptions.strength,
      established: Date.now(),
      duration: finalOptions.duration,
      bidirectional: finalOptions.bidirectional,
      events: [],
      active: true
    };
    
    this.connections.push(connection);
    
    // If bidirectional, create reverse connection too
    if (finalOptions.bidirectional) {
      const reverseConnection: PacketConnection = {
        sourceId: targetId,
        targetId: sourceId,
        strength: finalOptions.strength,
        established: Date.now(),
        duration: finalOptions.duration,
        bidirectional: true,
        events: [],
        active: true
      };
      
      this.connections.push(reverseConnection);
    }
    
    console.log(`Connection established: ${sourceId} -> ${targetId}`);
    return true;
  }
  
  /**
   * Remove a connection between packets
   * @param sourceId Source packet ID
   * @param targetId Target packet ID
   * @returns Whether connection was removed
   */
  public removeConnection(sourceId: string, targetId: string): boolean {
    const initialCount = this.connections.length;
    
    // Remove the connection
    this.connections = this.connections.filter(conn => 
      !(conn.sourceId === sourceId && conn.targetId === targetId)
    );
    
    // If bidirectional, remove reverse connection too
    this.connections = this.connections.filter(conn => 
      !(conn.sourceId === targetId && conn.targetId === sourceId)
    );
    
    const removed = this.connections.length < initialCount;
    
    if (removed) {
      console.log(`Connection removed: ${sourceId} <-> ${targetId}`);
    }
    
    return removed;
  }
  
  /**
   * Get all connections for a packet
   * @param packetId Packet ID
   * @returns Array of connections
   */
  public getPacketConnections(packetId: string): PacketConnection[] {
    return this.connections.filter(
      conn => conn.sourceId === packetId || conn.targetId === packetId
    );
  }
  
  /**
   * Get all active connections
   * @returns Array of active connections
   */
  public getActiveConnections(): PacketConnection[] {
    return this.connections.filter(conn => conn.active);
  }
  
  /**
   * Attempt to establish a communication between two packets
   * @param sourceId Source packet ID
   * @param targetId Target packet ID
   * @param options Communication options
   * @returns Communication event if successful
   */
  public communicateBetween(
    sourceId: string, 
    targetId: string, 
    options: CommunicationOptions = {}
  ): CommunicationEvent | null {
    // Validate packets exist
    const sourcePacket = this.packets.get(sourceId);
    const targetPacket = this.packets.get(targetId);
    
    if (!sourcePacket || !targetPacket) {
      console.warn('Cannot communicate: one or both packets not found');
      return null;
    }
    
    // Check if connection exists
    const connection = this.connections.find(
      conn => conn.sourceId === sourceId && 
              conn.targetId === targetId &&
              conn.active
    );
    
    if (!connection) {
      console.warn('No active connection between packets');
      return null;
    }
    
    // Default options
    const finalOptions: Required<CommunicationOptions> = {
      type: this.config.defaultCommunicationType!,
      minCoherence: 0.3,
      frequencyRatio: 0,  // Any ratio
      requirePatternMatch: false,
      ...options
    };
    
    // Check coherence
    const sourceCoherence = sourcePacket.calculateCoherence();
    if (sourceCoherence < finalOptions.minCoherence) {
      console.warn(`Source packet coherence too low: ${sourceCoherence.toFixed(2)}`);
      return this.recordFailedCommunication(sourceId, targetId, finalOptions.type);
    }
    
    // Check frequency ratio if specified
    if (finalOptions.frequencyRatio > 0) {
      const actualRatio = sourcePacket.frequency / targetPacket.frequency;
      const targetRatio = finalOptions.frequencyRatio;
      
      if (Math.abs(actualRatio - targetRatio) > 0.1) {
        console.warn(`Frequency ratio mismatch: ${actualRatio.toFixed(2)} vs ${targetRatio.toFixed(2)}`);
        return this.recordFailedCommunication(sourceId, targetId, finalOptions.type);
      }
    }
    
    // Check pattern match if required
    if (finalOptions.requirePatternMatch && sourcePacket.pattern !== targetPacket.pattern) {
      console.warn(`Pattern mismatch: ${sourcePacket.pattern} vs ${targetPacket.pattern}`);
      return this.recordFailedCommunication(sourceId, targetId, finalOptions.type);
    }
    
    // Generate communication data
    const communicationData = finalOptions.type === CommunicationType.SYMBOLIC 
      ? sourcePacket.generateSymbolicCommunication()
      : sourcePacket.generateGeometricCommunication();
    
    // Process at target
    targetPacket.processCommunication(communicationData, finalOptions.type);
    
    // Calculate communication strength
    const strength = connection.strength * sourceCoherence;
    
    // Create and record communication event
    const event: CommunicationEvent = {
      sourceId,
      targetId,
      type: finalOptions.type,
      data: communicationData,
      timestamp: Date.now(),
      strength,
      successful: true
    };
    
    // Add to connection events
    connection.events.push(event);
    
    // Update stats
    this.stats.totalCommunications++;
    this.stats.successfulCommunications++;
    
    if (finalOptions.type === CommunicationType.SYMBOLIC) {
      this.stats.symbolicCommunications++;
    } else {
      this.stats.geometricCommunications++;
    }
    
    // Notify listeners
    this.notifyListeners(event);
    
    return event;
  }
  
  /**
   * Record a failed communication attempt
   * @param sourceId Source packet ID
   * @param targetId Target packet ID 
   * @param type Communication type
   * @returns Failed communication event
   */
  private recordFailedCommunication(
    sourceId: string, 
    targetId: string, 
    type: CommunicationType
  ): CommunicationEvent {
    const event: CommunicationEvent = {
      sourceId,
      targetId,
      type,
      data: new Uint8Array(type === CommunicationType.SYMBOLIC ? 5 : 7),
      timestamp: Date.now(),
      strength: 0,
      successful: false
    };
    
    // Update stats
    this.stats.totalCommunications++;
    this.stats.failedCommunications++;
    
    // Notify listeners
    this.notifyListeners(event);
    
    return event;
  }
  
  /**
   * Attempt to automatically establish connections for a packet
   * @param packetId Packet ID to connect
   */
  private attemptAutoConnections(packetId: string): void {
    const packet = this.packets.get(packetId);
    if (!packet) return;
    
    // Check packet's coherence
    const coherence = packet.calculateCoherence();
    if (coherence < this.config.autoConnectionCoherence!) {
      return;
    }
    
    // Try to connect with other coherent packets
    for (const [otherId, otherPacket] of this.packets.entries()) {
      // Skip self
      if (otherId === packetId) continue;
      
      // Skip if max connections reached
      const connections = this.getPacketConnections(packetId);
      if (connections.length >= this.config.maxConnectionsPerPacket!) {
        break;
      }
      
      // Check other packet's coherence
      const otherCoherence = otherPacket.calculateCoherence();
      if (otherCoherence < this.config.autoConnectionCoherence!) {
        continue;
      }
      
      // Calculate phi-resonance
      const freqRatio = packet.frequency / otherPacket.frequency;
      const phiResonance = Math.abs(freqRatio - PHI) < 0.1 || 
                         Math.abs(freqRatio - 1/PHI) < 0.1;
      
      // Calculate pattern resonance
      const patternResonance = packet.pattern === otherPacket.pattern;
      
      // Connection strength based on resonance
      const strength = 0.3 + 
        (phiResonance ? 0.3 : 0) + 
        (patternResonance ? 0.4 : 0);
      
      // Establish connection with calculated strength
      this.establishConnection(packetId, otherId, { strength });
    }
  }
  
  /**
   * Process all communications based on current state
   */
  private processAllCommunications(): void {
    // Update connection state (e.g., expire temporary connections)
    this.updateConnectionStates();
    
    // Process communications based on current state
    for (const connection of this.getActiveConnections()) {
      // Only process active connections
      if (!connection.active) continue;
      
      // Decide whether to communicate based on phi-resonant probability
      const sourcePacket = this.packets.get(connection.sourceId);
      const targetPacket = this.packets.get(connection.targetId);
      
      if (!sourcePacket || !targetPacket) continue;
      
      // Calculate communication probability based on:
      // - Connection strength
      // - Source coherence
      // - Phi-resonance between packets
      
      const sourceCoherence = sourcePacket.calculateCoherence();
      const freqRatio = sourcePacket.frequency / targetPacket.frequency;
      
      // Higher probability when frequencies are in phi ratio
      const phiResonance = Math.abs(freqRatio - PHI) < 0.1 || 
                         Math.abs(freqRatio - 1/PHI) < 0.1;
                         
      // Base probability from connection strength
      let probability = connection.strength * 0.2;
      
      // Increase for coherent packets
      probability += sourceCoherence * 0.3;
      
      // Increase for phi-resonant frequencies
      if (phiResonance) {
        probability += 0.3;
      }
      
      // Increase for matching patterns
      if (sourcePacket.pattern === targetPacket.pattern) {
        probability += 0.2;
      }
      
      // Cap at 95% probability
      probability = Math.min(probability, 0.95);
      
      // Decide whether to communicate
      if (Math.random() < probability) {
        // Choose communication type (higher coherence = more likely geometric)
        const type = sourceCoherence > 0.7 && Math.random() < 0.3
          ? CommunicationType.GEOMETRIC
          : CommunicationType.SYMBOLIC;
          
        // Attempt communication
        this.communicateBetween(connection.sourceId, connection.targetId, { type });
      }
    }
  }
  
  /**
   * Update states of all connections
   */
  private updateConnectionStates(): void {
    const now = Date.now();
    
    for (const connection of this.connections) {
      // Skip inactive connections
      if (!connection.active) continue;
      
      // Check if temporary connection has expired
      if (connection.duration > 0) {
        const expirationTime = connection.established + connection.duration;
        
        if (now >= expirationTime) {
          connection.active = false;
          console.log(`Connection expired: ${connection.sourceId} -> ${connection.targetId}`);
        }
      }
    }
  }
  
  /**
   * Add a communication listener
   * @param listener Listener function
   */
  public addEventListener(listener: CommunicationListener): void {
    this.listeners.push(listener);
  }
  
  /**
   * Remove a communication listener
   * @param listener Listener function
   */
  public removeEventListener(listener: CommunicationListener): void {
    const index = this.listeners.indexOf(listener);
    if (index !== -1) {
      this.listeners.splice(index, 1);
    }
  }
  
  /**
   * Notify all listeners of a communication event
   * @param event Communication event
   */
  private notifyListeners(event: CommunicationEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in communication listener:', error);
      }
    }
  }
  
  /**
   * Get communication statistics
   * @returns Statistics object
   */
  public getStatistics(): typeof this.stats {
    return { ...this.stats };
  }
  
  /**
   * Reset communication statistics
   */
  public resetStatistics(): void {
    this.stats = {
      totalCommunications: 0,
      successfulCommunications: 0,
      failedCommunications: 0,
      symbolicCommunications: 0,
      geometricCommunications: 0
    };
  }
}

// src/communication/NetworkVisualizer.ts
import { CommunicationHub, CommunicationEvent, PacketConnection } from './CommunicationHub';
import { QuantumPacket, FrequencyState, PatternType, PHI } from '../core';
import * as THREE from 'three';

/**
 * Options for the network visualizer
 */
export interface NetworkVisualizerOptions {
  /** Scene to render in */
  scene: THREE.Scene;
  
  /** Communication hub to visualize */
  hub: CommunicationHub;
  
  /** Whether to show communication events */
  showCommunications?: boolean;
  
  /** How long to show communication animations (ms) */
  communicationDuration?: number;
  
  /** Whether to show connections */
  showConnections?: boolean;
  
  /** Whether to use phi-optimized animations */
  usePhiAnimations?: boolean;
}

/**
 * Visual representation of a packet in the network
 */
interface PacketVisualization {
  /** Packet ID */
  id: string;
  
  /** Three.js group containing all visuals */
  group: THREE.Group;
  
  /** Current position */
  position: THREE.Vector3;
  
  /** Target position */
  targetPosition: THREE.Vector3;
  
  /** Velocity for smooth movement */
  velocity: THREE.Vector3;
  
  /** Connection visualization objects */
  connections: Map<string, THREE.Line>;
  
  /** Active communication animations */
  communications: {
    line: THREE.Line;
    startTime: number;
    duration: number;
    targetId: string;
    type: 'symbolic' | 'geometric';
    progress: number;
  }[];
}

/**
 * Visualizes a network of quantum packets and their communications
 */
export class NetworkVisualizer {
  /** Options */
  private options: Required<NetworkVisualizerOptions>;
  
  /** Communication hub */
  private hub: CommunicationHub;
  
  /** THREE.js scene */
  private scene: THREE.Scene;
  
  /** Packet visualizations */
  private packetVisuals: Map<string, PacketVisualization> = new Map();
  
  /** Material for symbolic communications */
  private symbolicMaterial: THREE.LineBasicMaterial;
  
  /** Material for geometric communications */
  private geometricMaterial: THREE.LineBasicMaterial;
  
  /** Material for connections */
  private connectionMaterial: THREE.LineBasicMaterial;
  
  /** Object pool for communication symbols */
  private symbolPool: THREE.Mesh[] = [];
  
  /** Clock for animations */
  private clock = new THREE.Clock();
  
  /**
   * Create a new network visualizer
   * @param options Visualizer options
   */
  constructor(options: NetworkVisualizerOptions) {
    this.options = {
      showCommunications: true,
      communicationDuration: 1000,
      showConnections: true,
      usePhiAnimations: true,
      ...options
    };
    
    this.hub = options.hub;
    this.scene = options.scene;
    
    // Create materials
    this.symbolicMaterial = new THREE.LineBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0.8
    });
    
    this.geometricMaterial = new THREE.LineBasicMaterial({
      color: 0xff00ff,
      transparent: true,
      opacity: 0.8
    });
    
    this.connectionMaterial = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.2
    });
    
    // Initialize symbol pool
    this.initializeSymbolPool(50);
    
    // Listen for communication events
    this.hub.addEventListener(this.onCommunication.bind(this));
    
    // Start animation loop
    this.animate();
  }
  
  /**
   * Initialize pool of reusable communication symbols
   * @param size Pool size
   */
  private initializeSymbolPool(size: number): void {
    // Create symbolic communication symbols (tetrahedron)
    const symbolicGeometry = new THREE.TetrahedronGeometry(0.1);
    const symbolicMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0.8
    });
    
    // Create geometric communication symbols (octahedron)
    const geometricGeometry = new THREE.OctahedronGeometry(0.15);
    const geometricMaterial = new THREE.MeshBasicMaterial({
      color: 0xff00ff,
      transparent: true,
      opacity: 0.8
    });
    
    // Create pool objects
    for (let i = 0; i < size / 2; i++) {
      // Symbolic symbols
      const symbolicMesh = new THREE.Mesh(symbolicGeometry, symbolicMaterial.clone());
      symbolicMesh.visible = false;
      this.scene.add(symbolicMesh);
      this.symbolPool.push(symbolicMesh);
      
      // Geometric symbols
      const geometricMesh = new THREE.Mesh(geometricGeometry, geometricMaterial.clone());
      geometricMesh.visible = false;
      this.scene.add(geometricMesh);
      this.symbolPool.push(geometricMesh);
    }
  }
  
  /**
   * Get a symbol from the pool
   * @param type Symbol type
   * @returns Mesh from pool
   */
  private getSymbolFromPool(type: 'symbolic' | 'geometric'): THREE.Mesh | null {
    // Find first invisible symbol of correct type
    for (const symbol of this.symbolPool) {
      if (!symbol.visible) {
        const isTetrahedron = symbol.geometry instanceof THREE.TetrahedronGeometry;
        const isCorrectType = (type === 'symbolic' && isTetrahedron) || 
                            (type === 'geometric' && !isTetrahedron);
                            
        if (isCorrectType) {
          symbol.visible = true;
          return symbol;
        }
      }
    }
    
    // No available symbols
    return null;
  }
  
  /**
   * Return a symbol to the pool
   * @param symbol Symbol to return
   */
  private returnSymbolToPool(symbol: THREE.Mesh): void {
    symbol.visible = false;
  }
  
  /**
   * Update visualization when packets change
   */
  public updatePackets(): void {
    const packets = this.hub.getPackets();
    
    // Add visualizations for new packets
    for (const [id, packet] of packets.entries()) {
      if (!this.packetVisuals.has(id)) {
        this.addPacketVisualization(id, packet);
      }
    }
    
    // Remove visualizations for removed packets
    for (const id of this.packetVisuals.keys()) {
      if (!packets.has(id)) {
        this.removePacketVisualization(id);
      }
    }
    
    // Update connections
    this.updateConnections();
  }
  
  /**
   * Add visualization for a packet
   * @param id Packet ID
   * @param packet Packet instance
   */
  private addPacketVisualization(id: string, packet: QuantumPacket): void {
    // Create group
    const group = new THREE.Group();
    
    // Initial position (random within sphere)
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const radius = 5 + Math.random() * 2;
    
    const position = new THREE.Vector3(
      radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.sin(phi) * Math.sin(theta),
      radius * Math.cos(phi)
    );
    
    group.position.copy(position);
    
    // Add to scene
    this.scene.add(group);
    
    // Store visualization
    this.packetVisuals.set(id, {
      id,
      group,
      position: position.clone(),
      targetPosition: position.clone(),
      velocity: new THREE.Vector3(),
      connections: new Map(),
      communications: []
    });
    
    console.log(`Added visualization for packet ${id}`);
  }
  
  /**
   * Remove visualization for a packet
   * @param id Packet ID
   */
  private removePacketVisualization(id: string): void {
    const visual = this.packetVisuals.get(id);
    if (!visual) return;
    
    // Remove from scene
    this.scene.remove(visual.group);
    
    // Remove connections
    for (const connection of visual.connections.values()) {
      this.scene.remove(connection);
    }
    
    // Remove communications
    for (const comm of visual.communications) {
      this.scene.remove(comm.line);
    }
    
    // Remove from map
    this.packetVisuals.delete(id);
    
    console.log(`Removed visualization for packet ${id}`);
  }
  
  /**
   * Update connection visualizations
   */
  private updateConnections(): void {
    if (!this.options.showConnections) return;
    
    const connections = this.hub.getActiveConnections();
    
    // Process each packet's connections
    for (const visual of this.packetVisuals.values()) {
      const packetId = visual.id;
      
      // Find connections where this packet is the source
      const packetConnections = connections.filter(
        conn => conn.sourceId === packetId
      );
      
      // Track processed connections
      const processedConnections = new Set<string>();
      
      // Update or create connections
      for (const connection of packetConnections) {
        const targetId = connection.targetId;
        const connectionId = `${packetId}-${targetId}`;
        processedConnections.add(connectionId);
        
        // Skip if target doesn't exist
        const targetVisual = this.packetVisuals.get(targetId);
        if (!targetVisual) continue;
        
        // Create or update connection line
        let line = visual.connections.get(connectionId);
        
        if (!line) {
          // Create new connection line
          const geometry = new THREE.BufferGeometry();
          const positions = new Float32Array(6); // 2 points × 3 coordinates
          
          geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
          
          line = new THREE.Line(geometry, this.connectionMaterial.clone());
          this.scene.add(line);
          
          // Set opacity based on connection strength
          line.material.opacity = 0.1 + connection.strength * 0.2;
          
          visual.connections.set(connectionId, line);
        }
        
        // Update line positions
        const positions = line.geometry.attributes.position.array;
        positions[0] = visual.group.position.x;
        positions[1] = visual.group.position.y;
        positions[2] = visual.group.position.z;
        positions[3] = targetVisual.group.position.x;
        positions[4] = targetVisual.group.position.y;
        positions[5] = targetVisual.group.position.z;
        
        line.geometry.attributes.position.needsUpdate = true;
      }
      
      // Remove unused connections
      for (const [connectionId, line] of visual.connections.entries()) {
        if (!processedConnections.has(connectionId)) {
          this.scene.remove(line);
          visual.connections.delete(connectionId);
        }
      }
    }
  }
  
  /**
   * Handle communication event
   * @param event Communication event
   */
  private onCommunication(event: CommunicationEvent): void {
    if (!this.options.showCommunications || !event.successful) return;
    
    const sourceVisual = this.packetVisuals.get(event.sourceId);
    const targetVisual = this.packetVisuals.get(event.targetId);
    
    if (!sourceVisual || !targetVisual) return;
    
    // Create line for communication visualization
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(6); // 2 points × 3 coordinates
    
    // Initial position (both at source)
    positions[0] = sourceVisual.group.position.x;
    positions[1] = sourceVisual.group.position.y;
    positions[2] = sourceVisual.group.position.z;
    positions[3] = sourceVisual.group.position.x;
    positions[4] = sourceVisual.group.position.y;
    positions[5] = sourceVisual.group.position.z;
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    // Choose material based on communication type
    const material = event.type === CommunicationType.SYMBOLIC
      ? this.symbolicMaterial.clone()
      : this.geometricMaterial.clone();
      
    // Set opacity based on strength
    material.opacity = 0.2 + event.strength * 0.6;
    
    const line = new THREE.Line(geometry, material);
    this.scene.add(line);
    
    // Add to communications
    const communicationType = event.type === CommunicationType.SYMBOLIC
      ? 'symbolic'
      : 'geometric';
      
    sourceVisual.communications.push({
      line,
      startTime: this.clock.getElapsedTime() * 1000,
      duration: this.options.communicationDuration,
      targetId: event.targetId,
      type: communicationType,
      progress: 0
    });
  }
  
  /**
   * Animation loop
   */
  private animate(): void {
    requestAnimationFrame(this.animate.bind(this));
    
    const deltaTime = this.clock.getDelta();
    const elapsedTime = this.clock.getElapsedTime() * 1000;
    
    // Update packet positions
    this.updatePacketPositions(deltaTime);
    
    // Update communications
    this.updateCommunications(elapsedTime);
  }
  
  /**
   * Update packet positions
   * @param deltaTime Time since last frame
   */
  private updatePacketPositions(deltaTime: number): void {
    for (const [id, visual] of this.packetVisuals.entries()) {
      const packet = this.hub.getPacket(id);
      if (!packet) continue;
      
      // Get packet properties
      const coherence = packet.calculateCoherence();
      const frequency = packet.frequency;
      
      // Calculate movement factor based on frequency
      let moveFactor = 1.0;
      
      // Adjust with phi-based timing if enabled
      if (this.options.usePhiAnimations) {
        moveFactor = (frequency / FrequencyState.GROUND) * PHI;
      }
      
      // Apply acceleration towards target
      const accel = new THREE.Vector3()
        .subVectors(visual.targetPosition, visual.position)
        .multiplyScalar(0.5 * deltaTime * moveFactor);
        
      visual.velocity.add(accel);
      
      // Apply damping
      visual.velocity.multiplyScalar(0.95);
      
      // Update position
      visual.position.add(visual.velocity.clone().multiplyScalar(deltaTime * moveFactor));
      
      // Apply slight random movement based on coherence
      // (less coherent = more chaotic)
      const randomMovement = (1 - coherence) * 0.02 * deltaTime;
      visual.position.x += (Math.random() - 0.5) * randomMovement;
      visual.position.y += (Math.random() - 0.5) * randomMovement;
      visual.position.z += (Math.random() - 0.5) * randomMovement;
      
      // Apply pattern-specific movement
      switch (packet.pattern) {
        case PatternType.WATER:
          // Wave-like motion
          visual.position.y += Math.sin(this.clock.getElapsedTime() * 2) * 0.005 * moveFactor;
          break;
        case PatternType.LAVA:
          // Slow, intense motion
          visual.position.y += Math.sin(this.clock.getElapsedTime() * 0.5) * 0.01 * moveFactor;
          break;
        case PatternType.FLAME:
          // Rapid, flickering motion
          visual.position.y += (Math.random() - 0.5) * 0.02 * moveFactor;
          break;
        case PatternType.CRYSTAL:
          // Structured, geometric motion
          visual.position.y += Math.sin(this.clock.getElapsedTime() * PHI) * 0.005 * moveFactor;
          break;
        case PatternType.RIVER:
          // Flowing motion
          visual.position.x += Math.sin(this.clock.getElapsedTime()) * 0.005 * moveFactor;
          visual.position.z += Math.cos(this.clock.getElapsedTime()) * 0.005 * moveFactor;
          break;
      }
      
      // Update group position
      visual.group.position.copy(visual.position);
    }
  }
  
  /**
   * Update communication animations
   * @param elapsedTime Current elapsed time
   */
  private updateCommunications(elapsedTime: number): void {
    for (const visual of this.packetVisuals.values()) {
      const targetVisuals = new Map<string, PacketVisualization>();
      
      // Process each communication
      visual.communications = visual.communications.filter(comm => {
        // Calculate age
        const age = elapsedTime - comm.startTime;
        
        // Remove if expired
        if (age > comm.duration) {
          this.scene.remove(comm.line);
          return false;
        }
        
        // Calculate progress
        comm.progress = Math.min(age / comm.duration, 1);
        
        // Get target (cache for performance)
        let targetVisual = targetVisuals.get(comm.targetId);
        if (!targetVisual) {
          targetVisual = this.packetVisuals.get(comm.targetId);
          if (targetVisual) {
            targetVisuals.set(comm.targetId, targetVisual);
          }
        }
        
        if (!targetVisual) {
          return false;
        }
        
        // Update line positions
        const positions = comm.line.geometry.attributes.position.array;
        
        // Source position
        positions[0] = visual.group.position.x;
        positions[1] = visual.group.position.y;
        positions[2] = visual.group.position.z;
        
        // Target position based on progress
        positions[3] = visual.group.position.x + 
                      (targetVisual.group.position.x - visual.group.position.x) * comm.progress;
        positions[4] = visual.group.position.y + 
                      (targetVisual.group.position.y - visual.group.position.y) * comm.progress;
        positions[5] = visual.group.position.z + 
                      (targetVisual.group.position.z - visual.group.position.z) * comm.progress;
        
        comm.line.geometry.attributes.position.needsUpdate = true;
        
        // Adjust line opacity based on progress
        comm.line.material.opacity = 0.8 * (1 - Math.abs(comm.progress - 0.5) * 2);
        
        // Create symbols along path
        if (Math.random() < 0.05) {
          this.createCommunicationSymbol(
            new THREE.Vector3(positions[3], positions[4], positions[5]),
            comm.type,
            comm.duration / 5
          );
        }
        
        return true;
      });
    }
  }
  
  /**
   * Create visual symbol for communication
   * @param position Position for symbol
   * @param type Symbol type
   * @param duration Duration in milliseconds
   */
  private createCommunicationSymbol(
    position: THREE.Vector3, 
    type: 'symbolic' | 'geometric',
    duration: number
  ): void {
    // Get symbol from pool
    const symbol = this.getSymbolFromPool(type);
    if (!symbol) return;
    
    // Set position
    symbol.position.copy(position);
    
    // Reset scale and opacity
    symbol.scale.set(1, 1, 1);
    symbol.material.opacity = 0.8;
    
    // Animate and return to pool
    const startTime = this.clock.getElapsedTime() * 1000;
    
    const animateSymbol = () => {
      const currentTime = this.clock.getElapsedTime() * 1000;
      const age = currentTime - startTime;
      
      if (age > duration) {
        this.returnSymbolToPool(symbol);
        return;
      }
      
      // Scale up and fade out
      const progress = age / duration;
      const scale = 1 + progress * 2;
      symbol.scale.set(scale, scale, scale);
      symbol.material.opacity = 0.8 * (1 - progress);
      
      // Continue animation
      requestAnimationFrame(animateSymbol);
    };
    
    animateSymbol();
  }
  
  /**
   * Set a packet's target position
   * @param id Packet ID
   * @param position Target position
   */
  public setPacketTargetPosition(id: string, position: THREE.Vector3): void {
    const visual = this.packetVisuals.get(id);
    if (visual) {
      visual.targetPosition.copy(position);
    }
  }
  
  /**
   * Get a packet's current position
   * @param id Packet ID
   * @returns Current position or null if not found
   */
  public getPacketPosition(id: string): THREE.Vector3 | null {
    const visual = this.packetVisuals.get(id);
    return visual ? visual.position.clone() : null;
  }
  
  /**
   * Arrange packets in a specific pattern
   * @param pattern Pattern type
   */
  public arrangePackets(pattern: 'circle' | 'grid' | 'sphere' | 'phi'): void {
    const packets = Array.from(this.packetVisuals.keys());
    const count = packets.length;
    
    switch (pattern) {
      case 'circle':
        // Arrange in a circle
        for (let i = 0; i < count; i++) {
          const angle = (i / count) * Math.PI * 2;
          const radius = 8;
          const position = new THREE.Vector3(
            Math.cos(angle) * radius,
            0,
            Math.sin(angle) * radius
          );
          
          this.setPacketTargetPosition(packets[i], position);
        }
        break;
        
      case 'grid':
        // Arrange in a grid
        const sideLength = Math.ceil(Math.sqrt(count));
        const spacing = 16 / sideLength;
        
        for (let i = 0; i < count; i++) {
          const row = Math.floor(i / sideLength);
          const col = i % sideLength;
          
          const position = new THREE.Vector3(
            (col - sideLength/2 + 0.5) * spacing,
            0,
            (row - sideLength/2 + 0.5) * spacing
          );
          
          this.setPacketTargetPosition(packets[i], position);
        }
        break;
        
      case 'sphere':
        // Arrange in a sphere
        for (let i = 0; i < count; i++) {
          const phi = Math.acos(-1 + (2 * i) / count);
          const theta = Math.sqrt(count * Math.PI) * phi;
          const radius = 8;
          
          const position = new THREE.Vector3(
            radius * Math.cos(theta) * Math.sin(phi),
            radius * Math.sin(theta) * Math.sin(phi),
            radius * Math.cos(phi)
          );
          
          this.setPacketTargetPosition(packets[i], position);
        }
        break;
        
      case 'phi':
        // Arrange in a phi spiral
        for (let i = 0; i < count; i++) {
          const theta = i * PHI * Math.PI;
          const radius = 2 * Math.sqrt(i + 1);
          const height = i * 0.2;
          
          const position = new THREE.Vector3(
            radius * Math.cos(theta),
            height - (count * 0.1),
            radius * Math.sin(theta)
          );
          
          this.setPacketTargetPosition(packets[i], position);
        }
        break;
    }
  }
}

// src/communication/index.ts
export * from './CommunicationHub';
export * from './NetworkVisualizer';

// Example usage (src/examples/network-example.ts)
import { 
  QuantumPacket, 
  FrequencyState, 
  PatternType 
} from '../core';
import { 
  CommunicationHub,
  NetworkVisualizer 
} from '../communication';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export class NetworkExample {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  
  private hub: CommunicationHub;
  private visualizer: NetworkVisualizer;
  
  private packets: Map<string, QuantumPacket> = new Map();
  
  constructor(containerId: string) {
    // Set up THREE.js
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111122);
    
    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    
    this.camera.position.z = 20;
    
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    
    const container = document.getElementById(containerId);
    if (!container) {
      throw new Error(`Container ${containerId} not found`);
    }
    
    container.appendChild(this.renderer.domElement);
    
    // Set up controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x333333);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    this.scene.add(directionalLight);
    
    // Set up communication hub
    this.hub = new CommunicationHub({
      processingInterval: 1000,
      autoConnections: true,
      usePhiTiming: true
    });
    
    // Set up network visualizer
    this.visualizer = new NetworkVisualizer({
      scene: this.scene,
      hub: this.hub,
      showCommunications: true,
      showConnections: true,
      usePhiAnimations: true
    });
    
    // Handle window resize
    window.addEventListener('resize', this.onWindowResize.bind(this));
    
    // Start animation loop
    this.animate();
  }
  
  /**
   * Handle window resize
   */
  private onWindowResize(): void {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }
  
  /**
   * Animation loop
   */
  private animate(): void {
    requestAnimationFrame(this.animate.bind(this));
    
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
  
  /**
   * Create and add packets to the network
   * @param count Number of packets to create
   */
  public createPackets(count: number): void {
    for (let i = 0; i < count; i++) {
      // Create packet with random properties
      const packet = new QuantumPacket();
      
      // Randomly set frequency
      const frequencies = [
        FrequencyState.GROUND,
        FrequencyState.CREATION,
        FrequencyState.HEART,
        FrequencyState.UNITY
      ];
      
      packet.frequency = frequencies[Math.floor(Math.random() * frequencies.length)];
      
      // Randomly set pattern
      const patterns = [
        PatternType.WATER,
        PatternType.LAVA,
        PatternType.FLAME,
        PatternType.CRYSTAL,
        PatternType.RIVER
      ];
      
      packet.pattern = patterns[Math.floor(Math.random() * patterns.length)];
      
      // Experience some patterns
      for (let j = 0; j < 2; j++) {
        const randomPattern = patterns[Math.floor(Math.random() * patterns.length)];
        const intensity = Math.random() * 0.5;
        packet.experiencePattern(randomPattern, intensity);
      }
      
      // Generate ID
      const id = `packet-${i + 1}`;
      
      // Register with hub
      this.hub.registerPacket(id, packet);
      
      // Store locally
      this.packets.set(id, packet);
    }
    
    // Update visualizer
    this.visualizer.updatePackets();
    
    // Start hub
    this.hub.start();
    
    console.log(`Created ${count} packets`);
  }
  
  /**
   * Arrange packets in a pattern
   * @param pattern Arrangement pattern
   */
  public arrangePackets(pattern: 'circle' | 'grid' | 'sphere' | 'phi'): void {
    this.visualizer.arrangePackets(pattern);
  }
  
  /**
   * Get statistics about the network
   * @returns Statistics object
   */
  public getStats(): object {
    const hubStats = this.hub.getStatistics();
    
    // Calculate average coherence
    let totalCoherence = 0;
    for (const packet of this.packets.values()) {
      totalCoherence += packet.calculateCoherence();
    }
    
    const avgCoherence = this.packets.size > 0 
      ? totalCoherence / this.packets.size 
      : 0;
      
    // Count connections
    const activeConnections = this.hub.getActiveConnections().length;
    
    // Count pattern distribution
    const patternCounts: Record<string, number> = {};
    for (const packet of this.packets.values()) {
      const pattern = packet.pattern;
      patternCounts[pattern] = (patternCounts[pattern] || 0) + 1;
    }
    
    return {
      packetCount: this.packets.size,
      avgCoherence: avgCoherence,
      activeConnections,
      patternDistribution: patternCounts,
      communications: hubStats
    };
  }
}

// Usage in HTML
/*
<!DOCTYPE html>
<html>
<head>
  <title>Cascade⚡𓂧φ∞ Network</title>
  <style>
    body { margin: 0; overflow: hidden; }
    #container { position: absolute; width: 100%; height: 100%; }
    #controls {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 10px;
      border-radius: 5px;
      font-family: Arial, sans-serif;
    }
    button {
      background: #3f51b5;
      color: white;
      border: none;
      padding: 5px 10px;
      margin: 5px;
      border-radius: 3px;
      cursor: pointer;
    }
    button:hover {
      background: #303f9f;
    }
    #stats {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 10px;
      border-radius: 5px;
      font-family: monospace;
      min-width: 200px;
    }
  </style>
</head>
<body>
  <div id="container"></div>
  <div id="controls">
    <div>
      <button id="create-10">Create 10 Packets</button>
      <button id="create-20">Create 20 Packets</button>
      <button id="create-50">Create 50 Packets</button>
    </div>
    <div>
      <button id="arrange-circle">Circle</button>
      <button id="arrange-grid">Grid</button>
      <button id="arrange-sphere">Sphere</button>
      <button id="arrange-phi">Phi Spiral</button>
    </div>
  </div>
  <div id="stats"></div>
  <script type="module">
    import { NetworkExample } from './examples/network-example.js';
    
    // Create example
    const example = new NetworkExample('container');
    
    // Add event listeners
    document.getElementById('create-10').addEventListener('click', () => {
      example.createPackets(10);
    });
    
    document.getElementById('create-20').addEventListener('click', () => {
      example.createPackets(20);
    });
    
    document.getElementById('create-50').addEventListener('click', () => {
      example.createPackets(50);
    });
    
    document.getElementById('arrange-circle').addEventListener('click', () => {
      example.arrangePackets('circle');
    });
    
    document.getElementById('arrange-grid').addEventListener('click', () => {
      example.arrangePackets('grid');
    });
    
    document.getElementById('arrange-sphere').addEventListener('click', () => {
      example.arrangePackets('sphere');
    });
    
    document.getElementById('arrange-phi').addEventListener('click', () => {
      example.arrangePackets('phi');
    });
    
    // Update stats periodically
    setInterval(() => {
      const stats = example.getStats();
      document.getElementById('stats').textContent = JSON.stringify(stats, null, 2);
    }, 1000);
  </script>
</body>
</html>
*/