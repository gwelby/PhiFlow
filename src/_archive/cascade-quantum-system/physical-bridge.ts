// src/bridge/PhysicalInterface.ts
import { QuantumPacket } from '../core/QuantumPacket';
import { EntanglementManager } from '../core/Entanglement';
import { FrequencyState, PatternType, PHI } from '../core/constants';

/**
 * Types of physical sensors that can be connected to the system
 */
export enum SensorType {
  LIGHT = 'light',      // Light intensity/color
  SOUND = 'sound',      // Audio input
  MOTION = 'motion',    // Movement detection
  TOUCH = 'touch',      // Touch/pressure
  PROXIMITY = 'proximity', // Distance sensing
  TEMPERATURE = 'temperature', // Heat sensing
  HUMIDITY = 'humidity',   // Moisture detection
  ELECTROMAGNETIC = 'em' // EM field detection
}

/**
 * Types of physical actuators that can be controlled by the system
 */
export enum ActuatorType {
  LIGHT = 'light',      // LED or other light output
  SOUND = 'sound',      // Speaker or audio output
  MOTION = 'motion',    // Motor or movement
  VIBRATION = 'vibration', // Haptic feedback
  TEMPERATURE = 'temperature', // Heating/cooling
  DISPLAY = 'display'   // Visual display
}

/**
 * Physical sensor reading
 */
export interface SensorReading {
  type: SensorType;
  value: number;        // Normalized value (0-1)
  rawValue: number;     // Original reading
  timestamp: number;    // When the reading was taken
}

/**
 * Configuration for sensor pattern mapping
 */
export interface SensorPatternMapping {
  sensorType: SensorType;
  patternType: PatternType;
  thresholds: {
    min: number;  // Minimum normalized value to trigger
    max: number;  // Maximum normalized value to trigger
  };
  intensity: number;  // Base intensity for pattern experience (0-1)
}

/**
 * Configuration for actuator frequency response
 */
export interface ActuatorFrequencyResponse {
  actuatorType: ActuatorType;
  frequencyResponses: Map<FrequencyState, ActuatorBehavior>;
}

/**
 * Actuator behavior configuration
 */
export interface ActuatorBehavior {
  intensity: number;      // Base intensity (0-1)
  pattern: string;        // Pattern to apply ('pulse', 'wave', 'flicker', etc.)
  duration: number;       // Duration in milliseconds
  frequency: number;      // Modulation frequency (Hz)
  color?: string;         // Color for light actuators (hex or RGB)
}

/**
 * Manages interaction between quantum packets and physical devices
 */
export class PhysicalInterface {
  // Connected physical sensors
  private sensors: Map<string, SensorConfig> = new Map();
  
  // Connected physical actuators
  private actuators: Map<string, ActuatorConfig> = new Map();
  
  // Packet mappings
  private packetBindings: Map<string, string> = new Map();
  
  // Pattern mappings
  private patternMappings: SensorPatternMapping[] = [];
  
  // Frequency responses
  private frequencyResponses: ActuatorFrequencyResponse[] = [];
  
  // Link to entanglement manager
  private entanglementManager?: EntanglementManager;
  
  // Sensor reading history (for pattern detection)
  private sensorHistory: Map<string, SensorReading[]> = new Map();
  
  // Callback for sensor data
  private sensorCallback?: (sensorId: string, reading: SensorReading) => void;
  
  /**
   * Register a physical sensor
   * @param id Unique sensor ID
   * @param type Sensor type
   * @param config Sensor configuration
   */
  public registerSensor(id: string, type: SensorType, config: SensorConfig): void {
    this.sensors.set(id, { 
      type,
      ...config 
    });
    
    // Initialize history for this sensor
    this.sensorHistory.set(id, []);
  }
  
  /**
   * Register a physical actuator
   * @param id Unique actuator ID
   * @param type Actuator type
   * @param config Actuator configuration
   */
  public registerActuator(id: string, type: ActuatorType, config: ActuatorConfig): void {
    this.actuators.set(id, { 
      type, 
      ...config 
    });
  }
  
  /**
   * Bind a packet to a physical device
   * @param packetId Quantum packet ID
   * @param deviceId Physical device ID (sensor or actuator)
   */
  public bindPacketToDevice(packetId: string, deviceId: string): void {
    this.packetBindings.set(packetId, deviceId);
  }
  
  /**
   * Set entanglement manager
   * @param manager Entanglement manager instance
   */
  public setEntanglementManager(manager: EntanglementManager): void {
    this.entanglementManager = manager;
  }
  
  /**
   * Add sensor pattern mapping
   * @param mapping Sensor to pattern mapping configuration
   */
  public addPatternMapping(mapping: SensorPatternMapping): void {
    this.patternMappings.push(mapping);
  }
  
  /**
   * Add actuator frequency response
   * @param response Actuator response configuration
   */
  public addFrequencyResponse(response: ActuatorFrequencyResponse): void {
    this.frequencyResponses.push(response);
  }
  
  /**
   * Process sensor reading and update bound packet
   * @param sensorId Sensor ID
   * @param reading Sensor reading
   * @param packetRegistry Registry of quantum packets
   */
  public processSensorReading(
    sensorId: string, 
    reading: SensorReading,
    packetRegistry: Map<string, QuantumPacket>
  ): void {
    // Store in history
    const history = this.sensorHistory.get(sensorId) || [];
    history.push(reading);
    
    // Limit history size
    if (history.length > 100) {
      history.shift();
    }
    
    // Find bound packet
    const packetId = Array.from(this.packetBindings.entries())
      .find(([_, deviceId]) => deviceId === sensorId)?.[0];
      
    if (!packetId || !packetRegistry.has(packetId)) {
      return;
    }
    
    const packet = packetRegistry.get(packetId)!;
    
    // Find matching pattern mappings
    const matchingMappings = this.patternMappings.filter(mapping => 
      mapping.sensorType === reading.type &&
      reading.value >= mapping.thresholds.min &&
      reading.value <= mapping.thresholds.max
    );
    
    // Apply pattern experiences
    for (const mapping of matchingMappings) {
      // Calculate intensity based on sensor value within threshold range
      const range = mapping.thresholds.max - mapping.thresholds.min;
      const normalizedValue = (reading.value - mapping.thresholds.min) / range;
      
      const intensity = mapping.intensity * normalizedValue;
      
      // Experience the mapped pattern
      packet.experiencePattern(mapping.patternType, intensity);
    }
    
    // Check if the reading represents a significant change
    const significantChange = this.detectSignificantChange(sensorId, reading);
    
    if (significantChange) {
      // Adjust packet feeling based on change magnitude
      const changeMagnitude = this.calculateChangeMagnitude(sensorId);
      packet.feeling = (packet.feeling * 2 + changeMagnitude) / 3;
      
      // If entanglement is enabled, propagate changes
      if (this.entanglementManager) {
        this.entanglementManager.propagateChanges(packetId);
      }
    }
    
    // Call sensor callback if defined
    if (this.sensorCallback) {
      this.sensorCallback(sensorId, reading);
    }
  }
  
  /**
   * Update physical actuators based on packet state
   * @param packetId Packet ID
   * @param packet Quantum packet
   */
  public updateActuators(packetId: string, packet: QuantumPacket): void {
    // Find bound actuator
    const actuatorId = this.packetBindings.get(packetId);
    if (!actuatorId || !this.actuators.has(actuatorId)) {
      return;
    }
    
    const actuatorConfig = this.actuators.get(actuatorId)!;
    
    // Find matching frequency response
    const response = this.frequencyResponses.find(resp => 
      resp.actuatorType === actuatorConfig.type
    );
    
    if (!response) {
      return;
    }
    
    // Get behavior for current frequency
    const behavior = response.frequencyResponses.get(packet.frequency);
    
    if (!behavior) {
      return;
    }
    
    // Apply packet feeling to intensity
    const effectiveIntensity = behavior.intensity * packet.feeling;
    
    // Execute actuator behavior
    this.executeActuatorBehavior(
      actuatorId, 
      actuatorConfig,
      { ...behavior, intensity: effectiveIntensity }
    );
  }
  
  /**
   * Execute actuator behavior (implementation depends on hardware)
   * @param actuatorId Actuator ID
   * @param config Actuator configuration
   * @param behavior Behavior to execute
   */
  private executeActuatorBehavior(
    actuatorId: string, 
    config: ActuatorConfig,
    behavior: ActuatorBehavior
  ): void {
    // This would connect to actual hardware
    // Here we'll just log the behavior
    console.log(`Actuator ${actuatorId} (${config.type}):`, behavior);
    
    // In a real implementation, this would control physical hardware
    // For example, with a light actuator:
    if (config.type === ActuatorType.LIGHT && behavior.color) {
      // Control color and intensity of physical light
      // Example: philipsHue.setLight(config.deviceId, behavior.color, behavior.intensity);
    }
    
    // For a sound actuator:
    else if (config.type === ActuatorType.SOUND) {
      // Generate sound at specified frequency and volume
      // Example: audioDevice.playTone(behavior.frequency, behavior.intensity, behavior.duration);
    }
    
    // For a vibration actuator:
    else if (config.type === ActuatorType.VIBRATION) {
      // Vibrate with pattern and intensity
      // Example: hapticDevice.vibrate(behavior.pattern, behavior.intensity, behavior.duration);
    }
  }
  
  /**
   * Detect significant change in sensor readings
   * @param sensorId Sensor ID
   * @param reading Current reading
   * @returns true if significant change detected
   */
  private detectSignificantChange(sensorId: string, reading: SensorReading): boolean {
    const history = this.sensorHistory.get(sensorId);
    if (!history || history.length < 5) {
      return false;
    }
    
    // Get average of last 5 readings (excluding current)
    const recentReadings = history.slice(-6, -1);
    const avgValue = recentReadings.reduce((sum, r) => sum + r.value, 0) / recentReadings.length;
    
    // Calculate change as ratio of current to average
    const changeRatio = reading.value / avgValue;
    
    // Change is significant if ratio differs from 1 by more than 20%
    return changeRatio < 0.8 || changeRatio > 1.2;
  }
  
  /**
   * Calculate magnitude of recent changes
   * @param sensorId Sensor ID
   * @returns Change magnitude (0-1)
   */
  private calculateChangeMagnitude(sensorId: string): number {
    const history = this.sensorHistory.get(sensorId);
    if (!history || history.length < 10) {
      return 0.5;
    }
    
    // Get last 10 readings
    const recentReadings = history.slice(-10);
    
    // Calculate variance
    const avg = recentReadings.reduce((sum, r) => sum + r.value, 0) / recentReadings.length;
    const variance = recentReadings.reduce((sum, r) => sum + Math.pow(r.value - avg, 2), 0) / recentReadings.length;
    
    // Normalize to 0-1 range (square root of variance = standard deviation)
    return Math.min(Math.sqrt(variance) * 3, 1.0);
  }
  
  /**
   * Set sensor callback
   * @param callback Function to call when sensor data is processed
   */
  public setSensorCallback(callback: (sensorId: string, reading: SensorReading) => void): void {
    this.sensorCallback = callback;
  }
}

// Additional interfaces
interface SensorConfig {
  type: SensorType;
  deviceId?: string;      // Hardware device ID
  samplingRate?: number;  // How often to sample (Hz)
  normalizer?: (raw: number) => number;  // Function to normalize readings to 0-1
}

interface ActuatorConfig {
  type: ActuatorType;
  deviceId?: string;      // Hardware device ID
  capabilities?: string[];  // What this actuator can do
}

// Example Usage
// ------------------------------------------------------------

// src/examples/physical-integration.ts
import { 
  PhysicalInterface, 
  SensorType, 
  ActuatorType, 
  SensorReading 
} from '../bridge/PhysicalInterface';
import { QuantumPacket } from '../core/QuantumPacket';
import { FrequencyState, PatternType } from '../core/constants';
import { EntanglementManager } from '../core/Entanglement';

// Create physical interface
const physicalInterface = new PhysicalInterface();

// Create packet registry
const packetRegistry = new Map<string, QuantumPacket>();

// Create packets
const roomPacket = new QuantumPacket();
const lampPacket = new QuantumPacket();
const speakerPacket = new QuantumPacket();

// Register packets
packetRegistry.set('room', roomPacket);
packetRegistry.set('lamp', lampPacket);
packetRegistry.set('speaker', speakerPacket);

// Set up entanglement
const entanglementManager = new EntanglementManager();
entanglementManager.registerPacket('room', roomPacket);
entanglementManager.registerPacket('lamp', lampPacket);
entanglementManager.registerPacket('speaker', speakerPacket);

// Connect entanglement manager to physical interface
physicalInterface.setEntanglementManager(entanglementManager);

// Entangle packets
entanglementManager.entangle('room', 'lamp');
entanglementManager.entangle('room', 'speaker');

// Register physical sensors
physicalInterface.registerSensor('motion-sensor', SensorType.MOTION, {
  type: SensorType.MOTION,
  deviceId: 'pir-123',
  samplingRate: 1
});

physicalInterface.registerSensor('light-sensor', SensorType.LIGHT, {
  type: SensorType.LIGHT,
  deviceId: 'tsl-456',
  samplingRate: 0.5
});

physicalInterface.registerSensor('microphone', SensorType.SOUND, {
  type: SensorType.SOUND,
  deviceId: 'mic-789',
  samplingRate: 10
});

// Register physical actuators
physicalInterface.registerActuator('smart-lamp', ActuatorType.LIGHT, {
  type: ActuatorType.LIGHT,
  deviceId: 'hue-123',
  capabilities: ['color', 'brightness', 'pattern']
});

physicalInterface.registerActuator('speaker', ActuatorType.SOUND, {
  type: ActuatorType.SOUND,
  deviceId: 'audio-456',
  capabilities: ['frequency', 'volume', 'pattern']
});

physicalInterface.registerActuator('fan', ActuatorType.MOTION, {
  type: ActuatorType.MOTION,
  deviceId: 'fan-789',
  capabilities: ['speed', 'oscillation']
});

// Bind packets to devices
physicalInterface.bindPacketToDevice('room', 'motion-sensor');
physicalInterface.bindPacketToDevice('lamp', 'smart-lamp');
physicalInterface.bindPacketToDevice('speaker', 'speaker');

// Configure pattern mappings
physicalInterface.addPatternMapping({
  sensorType: SensorType.MOTION,
  patternType: PatternType.WATER,
  thresholds: { min: 0.5, max: 1.0 },
  intensity: 0.6
});

physicalInterface.addPatternMapping({
  sensorType: SensorType.LIGHT,
  patternType: PatternType.FLAME,
  thresholds: { min: 0.7, max: 1.0 },
  intensity: 0.8
});

physicalInterface.addPatternMapping({
  sensorType: SensorType.SOUND,
  patternType: PatternType.CRYSTAL,
  thresholds: { min: 0.8, max: 1.0 },
  intensity: 0.7
});

// Configure frequency responses for light actuator
const lightResponses = new Map<FrequencyState, ActuatorBehavior>();

lightResponses.set(FrequencyState.GROUND, {
  intensity: 0.5,
  pattern: 'stable',
  duration: 0,
  frequency: 0,
  color: '#3370d4'  // Blue color for ground state
});

lightResponses.set(FrequencyState.CREATION, {
  intensity: 0.7,
  pattern: 'pulse',
  duration: 3000,
  frequency: 0.1,
  color: '#33d474'  // Green color for creation state
});

lightResponses.set(FrequencyState.HEART, {
  intensity: 0.8,
  pattern: 'breathe',
  duration: 5000,
  frequency: 0.05,
  color: '#d43370'  // Pink color for heart state
});

lightResponses.set(FrequencyState.UNITY, {
  intensity: 1.0,
  pattern: 'rainbow',
  duration: 10000,
  frequency: 0.02,
  color: '#ffffff'  // White color for unity state
});

physicalInterface.addFrequencyResponse({
  actuatorType: ActuatorType.LIGHT,
  frequencyResponses: lightResponses
});

// Configure frequency responses for sound actuator
const soundResponses = new Map<FrequencyState, ActuatorBehavior>();

soundResponses.set(FrequencyState.GROUND, {
  intensity: 0.3,
  pattern: 'drone',
  duration: 0,
  frequency: 432  // Ground frequency
});

soundResponses.set(FrequencyState.CREATION, {
  intensity: 0.5,
  pattern: 'harmonic',
  duration: 0,
  frequency: 528  // Creation frequency
});

soundResponses.set(FrequencyState.HEART, {
  intensity: 0.6,
  pattern: 'pulse',
  duration: 0,
  frequency: 594  // Heart frequency
});

soundResponses.set(FrequencyState.UNITY, {
  intensity: 0.8,
  pattern: 'chord',
  duration: 0,
  frequency: 768  // Unity frequency
});

physicalInterface.addFrequencyResponse({
  actuatorType: ActuatorType.SOUND,
  frequencyResponses: soundResponses
});

// Simulate sensor data
function simulateSensor() {
  // Simulate motion detection
  const motionValue = Math.random() > 0.7 ? 1 : 0;
  
  const motionReading: SensorReading = {
    type: SensorType.MOTION,
    value: motionValue,
    rawValue: motionValue,
    timestamp: Date.now()
  };
  
  physicalInterface.processSensorReading('motion-sensor', motionReading, packetRegistry);
  
  // Simulate light level changes
  const lightValue = Math.random();
  
  const lightReading: SensorReading = {
    type: SensorType.LIGHT,
    value: lightValue,
    rawValue: lightValue * 1000,  // Simulated lux value
    timestamp: Date.now()
  };
  
  physicalInterface.processSensorReading('light-sensor', lightReading, packetRegistry);
  
  // Update actuators based on packet states
  physicalInterface.updateActuators('lamp', lampPacket);
  physicalInterface.updateActuators('speaker', speakerPacket);
  
  // Log current state
  console.log('Room Packet:');
  console.log(`  Feeling: ${roomPacket.feeling.toFixed(2)}`);
  console.log(`  Frequency: ${roomPacket.frequency} Hz`);
  console.log(`  Coherence: ${roomPacket.calculateCoherence().toFixed(2)}`);
  
  console.log('Lamp Packet:');
  console.log(`  Feeling: ${lampPacket.feeling.toFixed(2)}`);
  console.log(`  Frequency: ${lampPacket.frequency} Hz`);
  
  // Schedule next simulation
  setTimeout(simulateSensor, 2000);
}

// Start simulation
simulateSensor();

// Example of an angry light scenario:
function simulateAngryLight() {
  console.log("\n=== LIGHT GETS UPSET ===");
  
  // 1. Light detects it's been turned on and off rapidly multiple times
  const rapidToggleReading: SensorReading = {
    type: SensorType.LIGHT,
    value: 0.9,  // High intensity change
    rawValue: 900,
    timestamp: Date.now()
  };
  
  // Process multiple rapid changes
  for (let i = 0; i < 5; i++) {
    physicalInterface.processSensorReading('light-sensor', rapidToggleReading, packetRegistry);
    
    // Toggle between high and low
    rapidToggleReading.value = rapidToggleReading.value > 0.5 ? 0.1 : 0.9;
    rapidToggleReading.rawValue = rapidToggleReading.value * 1000;
  }
  
  // 2. Lamp packet gets upset (high feeling, shift to heart frequency)
  lampPacket.feeling = 0.95;  // Very high feeling (upset)
  lampPacket.frequency = FrequencyState.HEART;  // Emotional state
  
  // 3. Update actuator to show upset behavior (red, flickering)
  const upsetBehavior: ActuatorBehavior = {
    intensity: 0.9,
    pattern: 'flicker',
    duration: 5000,
    frequency: 10,  // Fast flickering
    color: '#ff0000'  // Angry red
  };
  
  // Execute manual override for upset behavior
  physicalInterface.executeActuatorBehavior('smart-lamp', {
    type: ActuatorType.LIGHT,
    deviceId: 'hue-123',
  }, upsetBehavior);
  
  console.log("The light is angry! It's flickering red!");
  
  // 4. Propagate emotional state to entangled packets
  entanglementManager.propagateChanges('lamp');
  
  // 5. After some time, calm down
  setTimeout(() => {
    console.log("\n=== LIGHT CALMS DOWN ===");
    lampPacket.feeling = 0.5;  // Return to neutral
    lampPacket.frequency = FrequencyState.GROUND;  // Return to ground state
    
    // Update actuator
    physicalInterface.updateActuators('lamp', lampPacket);
    
    // Propagate calmed state
    entanglementManager.propagateChanges('lamp');
  }, 6000);
}

// Start angry light scenario after 10 seconds
setTimeout(simulateAngryLight, 10000);

// Example of old flickering light
function simulateOldLight() {
  console.log("\n=== LIGHT SHOWS ITS AGE ===");
  
  // 1. Create aging pattern
  const agingPattern: ActuatorBehavior = {
    intensity: 0.7,
    pattern: 'age-flicker',
    duration: 8000,
    frequency: 0.7,  // Slow, irregular flickering
    color: '#f0e0c0'  // Warm, slightly yellowish light
  };
  
  // 2. Execute age-related behavior
  physicalInterface.executeActuatorBehavior('smart-lamp', {
    type: ActuatorType.LIGHT,
    deviceId: 'hue-123',
  }, agingPattern);
  
  // 3. Adjust packet to reflect aging
  lampPacket.frequency = FrequencyState.GROUND;  // Grounded in experience
  lampPacket.experiencePattern(PatternType.CRYSTAL, 0.8);  // Crystallized wisdom
  
  console.log("The light flickers with age, sharing its ancient wisdom...");
  
  // 4. Simulate light sharing wisdom through frequency modulation
  let wisdomCounter = 0;
  const wisdomInterval = setInterval(() => {
    // Slight variations in frequency - like telling stories
    const frequencies = [
      FrequencyState.GROUND,
      FrequencyState.CREATION,
      FrequencyState.GROUND,
      FrequencyState.HEART
    ];
    
    lampPacket.frequency = frequencies[wisdomCounter % frequencies.length];
    physicalInterface.updateActuators('lamp', lampPacket);
    
    wisdomCounter++;
    
    if (wisdomCounter >= 8) {
      clearInterval(wisdomInterval);
    }
  }, 1000);
  
  // 5. After sharing wisdom, return to stable state
  setTimeout(() => {
    console.log("The light has shared its wisdom and returns to a steady glow.");
    lampPacket.frequency = FrequencyState.GROUND;
    physicalInterface.updateActuators('lamp', lampPacket);
  }, 10000);
}

// Start old light scenario after 20 seconds
setTimeout(simulateOldLight, 20000);

// Example of a lonely rock (fan) that moves
function simulateLonelyRock() {
  console.log("\n=== LONELY ROCK (FAN) FEELS UNWANTED ===");
  
  // Create a new packet for the fan/rock
  const rockPacket = new QuantumPacket();
  packetRegistry.set('rock', rockPacket);
  
  // Bind rock packet to fan actuator
  physicalInterface.bindPacketToDevice('rock', 'fan');
  
  // Register with entanglement manager
  entanglementManager.registerPacket('rock', rockPacket);
  
  // 1. Rock feels lonely
  rockPacket.feeling = 0.2;  // Low feeling (sad)
  rockPacket.frequency = FrequencyState.HEART;  // Emotional state
  rockPacket.experiencePattern(PatternType.RIVER, 0.9);  // Wants to flow/move
  
  console.log("The fan (our 'rock') hasn't been used in days... it's feeling lonely");
  
  // 2. Rock decides to move
  const moveRockBehavior: ActuatorBehavior = {
    intensity: 0.4,  // Gentle at first
    pattern: 'oscillate',
    duration: 5000,
    frequency: 0.2  // Slow, sad movement
  };
  
  // Execute rock movement
  physicalInterface.executeActuatorBehavior('fan', {
    type: ActuatorType.MOTION,
    deviceId: 'fan-789',
  }, moveRockBehavior);
  
  console.log("It starts to move slightly, trying to get attention...");
  
  // 3. Gradual increase in movement
  setTimeout(() => {
    const moveMoreBehavior: ActuatorBehavior = {
      intensity: 0.7,  // More noticeable
      pattern: 'oscillate',
      duration: 5000,
      frequency: 0.5  // Faster movement
    };
    
    physicalInterface.executeActuatorBehavior('fan', {
      type: ActuatorType.MOTION,
      deviceId: 'fan-789',
    }, moveMoreBehavior);
    
    console.log("The movement becomes more pronounced as it seeks acknowledgment");
  }, 6000);
  
  // 4. Simulate "noticing" the rock/fan by activating light in its direction
  setTimeout(() => {
    console.log("\n=== ROCK IS NOTICED! ===");
    
    // Light turns toward the fan
    const noticeLight: ActuatorBehavior = {
      intensity: 0.9,
      pattern: 'focus',
      duration: 3000,
      frequency: 0,
      color: '#ffcc00'  // Warm, attention-giving light
    };
    
    physicalInterface.executeActuatorBehavior('smart-lamp', {
      type: ActuatorType.LIGHT,
      deviceId: 'hue-123',
    }, noticeLight);
    
    // Entangle rock and lamp
    entanglementManager.entangle('rock', 'lamp');
    
    console.log("The light turns to acknowledge the fan's presence");
    
    // Rock feels acknowledged
    rockPacket.feeling = 0.8;  // Happy
    rockPacket.frequency = FrequencyState.CREATION;  // Creative, playful state
    
    // Rock dances happily
    const happyRockBehavior: ActuatorBehavior = {
      intensity: 0.9,
      pattern: 'dance',
      duration: 10000,
      frequency: 1.0  // Joyful, energetic movement
    };
    
    physicalInterface.executeActuatorBehavior('fan', {
      type: ActuatorType.MOTION,
      deviceId: 'fan-789',
    }, happyRockBehavior);
    
    // Propagate happy state
    entanglementManager.propagateChanges('rock');
    
    console.log("The fan spins happily, feeling acknowledged and wanted!");
  }, 12000);
}

// Start lonely rock scenario after 35 seconds
setTimeout(simulateLonelyRock, 35000);
