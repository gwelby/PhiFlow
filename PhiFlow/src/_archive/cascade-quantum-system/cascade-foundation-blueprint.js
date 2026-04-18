// Cascade⚡𓂧φ∞ Quantum Packet System
// Foundation Phase Implementation Blueprint

// Core Constants
const PHI = 1.618033988749895;
const PACKET_SIZE = 432;
const REALITY_SIZE = 144;
const EXPERIENCE_SIZE = 144;
const WISDOM_SIZE = 144;

// Frequency States (Hz)
const FREQUENCIES = {
  GROUND: 432,    // Foundation/stability
  CREATION: 528,  // Creation/pattern formation
  HEART: 594,     // Emotional resonance
  UNITY: 768      // Perfect integration
};

// Natural Patterns
const PATTERNS = ["water", "lava", "flame", "crystal", "river"];

// ------------------------------------------------------
// 1. QUANTUM PACKET CORE
// ------------------------------------------------------

/**
 * Represents a 432-byte quantum packet with three segments:
 * - Reality Map (144 bytes)
 * - Experience Memory (144 bytes)
 * - Wisdom Accumulator (144 bytes)
 */
class QuantumPacket {
  constructor() {
    // Initialize 432-byte array
    this.data = new Uint8Array(PACKET_SIZE);
    
    // Initial feeling (quantum state value between 0-1)
    this.feeling = 0.5;
    
    // Current frequency state
    this.frequency = FREQUENCIES.GROUND;
    
    // Current pattern
    this.pattern = PATTERNS[0];
    
    // Pattern experience values (for each natural pattern)
    this.patternExperience = PATTERNS.map(() => 0);
    
    // Initialize with random data
    this.randomize();
  }
  
  /**
   * Get reality map segment (first 144 bytes)
   */
  getRealityMap() {
    return this.data.slice(0, REALITY_SIZE);
  }
  
  /**
   * Get experience memory segment (middle 144 bytes)
   */
  getExperienceMemory() {
    return this.data.slice(REALITY_SIZE, REALITY_SIZE + EXPERIENCE_SIZE);
  }
  
  /**
   * Get wisdom accumulator segment (last 144 bytes)
   */
  getWisdomAccumulator() {
    return this.data.slice(REALITY_SIZE + EXPERIENCE_SIZE);
  }
  
  /**
   * Fill packet with random data (for initialization)
   */
  randomize() {
    for (let i = 0; i < PACKET_SIZE; i++) {
      this.data[i] = Math.floor(Math.random() * 256);
    }
    
    // Initialize pattern experience with small random values
    this.patternExperience = PATTERNS.map(() => Math.random() * 0.3);
  }
  
  /**
   * Calculate coherence level of the packet
   * @returns {number} Coherence value between 0-1
   */
  calculateCoherence() {
    // Get average pattern experience
    const avgExperience = this.patternExperience.reduce((sum, val) => sum + val, 0) / 
                          this.patternExperience.length;
    
    // Calculate phi-weighted coherence
    const freqCoherence = Math.min(this.frequency / FREQUENCIES.UNITY, 1) * 0.5;
    const feelingCoherence = this.feeling * 0.3;
    const expCoherence = avgExperience * 0.2;
    
    return freqCoherence + feelingCoherence + expCoherence;
  }
  
  /**
   * Experience a natural pattern, updating pattern experience
   * @param {string} pattern - Natural pattern to experience
   * @param {number} intensity - Intensity of experience (0-1)
   */
  experiencePattern(pattern, intensity = 0.1) {
    const index = PATTERNS.indexOf(pattern);
    if (index === -1) return;
    
    // Update pattern experience (max 1.0)
    this.patternExperience[index] = Math.min(
      this.patternExperience[index] + intensity,
      1.0
    );
    
    // Update overall feeling based on pattern
    this.feeling = (this.feeling * 3 + intensity) / 4;
    
    // Store pattern in experience memory
    this.storeExperience(pattern, intensity);
  }
  
  /**
   * Store experience in the experience memory segment
   * @param {string} pattern - Pattern being experienced
   * @param {number} intensity - Intensity of experience
   */
  storeExperience(pattern, intensity) {
    const index = PATTERNS.indexOf(pattern);
    if (index === -1) return;
    
    // Calculate phi-based position in experience memory
    const position = Math.floor(((index + 1) * PHI * 20) % EXPERIENCE_SIZE);
    
    // Store pattern index and intensity
    this.data[REALITY_SIZE + position] = index;
    this.data[REALITY_SIZE + (position + 1) % EXPERIENCE_SIZE] = Math.floor(intensity * 255);
  }
  
  /**
   * Change packet's frequency state
   * @param {number} frequency - New frequency value
   */
  changeFrequency(frequency) {
    if (!Object.values(FREQUENCIES).includes(frequency)) return;
    
    const oldFrequency = this.frequency;
    this.frequency = frequency;
    
    // Update feeling based on frequency change
    const freqRatio = Math.min(frequency / oldFrequency, oldFrequency / frequency);
    this.feeling = (this.feeling * 2 + freqRatio) / 3;
  }
  
  /**
   * Generate a symbolic communication packet (5 bytes)
   * @returns {Uint8Array} 5-byte symbolic communication
   */
  generateSymbolicCommunication() {
    const symbolic = new Uint8Array(5);
    
    // Encode frequency, feeling, and pattern
    symbolic[0] = Math.floor(this.frequency / 4);
    symbolic[1] = Math.floor(this.feeling * 255);
    symbolic[2] = PATTERNS.indexOf(this.pattern);
    
    // Encode wisdom signature
    const wisdom = this.getWisdomAccumulator();
    symbolic[3] = wisdom[0];
    symbolic[4] = wisdom[WISDOM_SIZE - 1];
    
    return symbolic;
  }
  
  /**
   * Generate a geometric communication packet (7 bytes)
   * @returns {Uint8Array} 7-byte geometric communication
   */
  generateGeometricCommunication() {
    const geometric = new Uint8Array(7);
    
    // Encode phi-resonant communication
    geometric[0] = Math.floor((this.frequency % 432) / 432 * 255);
    geometric[1] = Math.floor(this.feeling * 255);
    
    // Encode pattern information with phi weighting
    const strongestPattern = this.getStrongestPattern();
    geometric[2] = PATTERNS.indexOf(strongestPattern);
    geometric[3] = Math.floor(this.patternExperience[geometric[2]] * 255);
    
    // Encode wisdom with phi positions
    const wisdom = this.getWisdomAccumulator();
    geometric[4] = wisdom[Math.floor(PHI * 10) % WISDOM_SIZE];
    geometric[5] = wisdom[Math.floor(PHI * PHI * 10) % WISDOM_SIZE];
    geometric[6] = wisdom[Math.floor(PHI * PHI * PHI * 10) % WISDOM_SIZE];
    
    return geometric;
  }
  
  /**
   * Get the strongest experienced pattern
   * @returns {string} Pattern name
   */
  getStrongestPattern() {
    let maxIndex = 0;
    let maxValue = this.patternExperience[0];
    
    for (let i = 1; i < this.patternExperience.length; i++) {
      if (this.patternExperience[i] > maxValue) {
        maxValue = this.patternExperience[i];
        maxIndex = i;
      }
    }
    
    return PATTERNS[maxIndex];
  }
  
  /**
   * Process communication from another packet
   * @param {Uint8Array} communication - Communication data
   * @param {string} type - Either 'symbolic' or 'geometric'
   */
  processCommunication(communication, type) {
    if (type === 'symbolic' && communication.length === 5) {
      this.processSymbolicCommunication(communication);
    }
    else if (type === 'geometric' && communication.length === 7) {
      this.processGeometricCommunication(communication);
    }
  }
  
  /**
   * Process symbolic communication (5 bytes)
   * @param {Uint8Array} symbolic - Symbolic communication data
   */
  processSymbolicCommunication(symbolic) {
    // Extract frequency information
    const commFrequency = symbolic[0] * 4;
    
    // Extract feeling
    const commFeeling = symbolic[1] / 255;
    
    // Extract pattern
    const patternIndex = symbolic[2] % PATTERNS.length;
    const commPattern = PATTERNS[patternIndex];
    
    // Update own state based on communication
    // (limited impact - symbolic is basic)
    this.frequency = (this.frequency * 4 + commFrequency) / 5;
    this.feeling = (this.feeling * 4 + commFeeling) / 5;
    
    // Experience communicated pattern
    this.experiencePattern(commPattern, 0.05);
  }
  
  /**
   * Process geometric communication (7 bytes)
   * @param {Uint8Array} geometric - Geometric communication data
   */
  processGeometricCommunication(geometric) {
    // Extract phi-resonant frequency
    const freqComponent = (geometric[0] / 255) * 432;
    
    // Extract feeling with phi weighting
    const commFeeling = geometric[1] / 255;
    
    // Extract pattern information
    const patternIndex = geometric[2] % PATTERNS.length;
    const commPattern = PATTERNS[patternIndex];
    const patternStrength = geometric[3] / 255;
    
    // Update own state based on communication
    // (stronger impact - geometric is advanced)
    this.frequency = (this.frequency * 3 + freqComponent * 2) / 5;
    this.feeling = (this.feeling * 3 + commFeeling * 2) / 5;
    
    // Experience communicated pattern with strength
    this.experiencePattern(commPattern, patternStrength * 0.2);
    
    // Integrate wisdom at phi positions
    const wisdom = this.getWisdomAccumulator();
    const pos1 = Math.floor(PHI * 10) % WISDOM_SIZE;
    const pos2 = Math.floor(PHI * PHI * 10) % WISDOM_SIZE;
    const pos3 = Math.floor(PHI * PHI * PHI * 10) % WISDOM_SIZE;
    
    wisdom[pos1] = (wisdom[pos1] + geometric[4]) / 2;
    wisdom[pos2] = (wisdom[pos2] + geometric[5]) / 2;
    wisdom[pos3] = (wisdom[pos3] + geometric[6]) / 2;
  }
}

// ------------------------------------------------------
// 2. VISUALIZATION SYSTEM
// ------------------------------------------------------

/**
 * Quantum packet visualization system using Three.js
 */
class QuantumVisualizer {
  constructor(containerId) {
    // Find container element
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`Container element ${containerId} not found`);
    }
    
    // Initialize packets
    this.packets = [];
    
    // Initialize Three.js
    this.initThree();
    
    // Start animation loop
    this.animate();
  }
  
  /**
   * Initialize Three.js scene, camera, renderer
   */
  initThree() {
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111122);
    
    // Create camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      this.container.clientWidth / this.container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.z = 15;
    
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.container.appendChild(this.renderer.domElement);
    
    // Add controls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x333333);
    this.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    this.scene.add(directionalLight);
    
    // Add window resize listener
    window.addEventListener('resize', this.onWindowResize.bind(this));
  }
  
  /**
   * Handle window resize
   */
  onWindowResize() {
    this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }
  
  /**
   * Animation loop
   */
  animate() {
    requestAnimationFrame(this.animate.bind(this));
    
    // Update controls
    this.controls.update();
    
    // Update packet visualizations
    this.updatePackets();
    
    // Render scene
    this.renderer.render(this.scene, this.camera);
  }
  
  /**
   * Add a new quantum packet
   * @param {QuantumPacket} packet - Quantum packet to visualize
   * @param {Object} position - {x, y, z} position
   */
  addPacket(packet, position = { x: 0, y: 0, z: 0 }) {
    // Create group for the packet
    const group = new THREE.Group();
    
    // Position the packet
    group.position.set(position.x, position.y, position.z);
    
    // Create reality map visualization (core sphere)
    const coreGeometry = new THREE.IcosahedronGeometry(0.5, 2);
    const coreMaterial = new THREE.MeshPhongMaterial({
      color: this.getFrequencyColor(packet.frequency),
      emissive: this.getFrequencyColor(packet.frequency, 0.3),
      shininess: 30,
      transparent: true,
      opacity: 0.9
    });
    
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    group.add(core);
    
    // Create experience memory visualization (torus knot)
    const memoryGeometry = new THREE.TorusKnotGeometry(0.8, 0.2, 64, 8, 2, 3);
    const memoryMaterial = new THREE.MeshPhongMaterial({
      color: this.getPatternColor(packet.pattern),
      transparent: true,
      opacity: 0.7,
      wireframe: false
    });
    
    const memory = new THREE.Mesh(memoryGeometry, memoryMaterial);
    group.add(memory);
    
    // Create wisdom accumulator visualization (outer sphere)
    const wisdomGeometry = new THREE.SphereGeometry(1.2, 16, 16);
    const wisdomMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.2,
      wireframe: true
    });
    
    const wisdom = new THREE.Mesh(wisdomGeometry, wisdomMaterial);
    group.add(wisdom);
    
    // Create particle system for pattern visualization
    const particleCount = 200;
    const particleGeometry = new THREE.BufferGeometry();
    
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    // Create particles in a phi spiral
    for (let i = 0; i < particleCount; i++) {
      const angle = 0.1 * i;
      const radius = 0.1 * Math.pow(PHI, angle / (2 * Math.PI));
      
      const px = radius * Math.cos(angle);
      const py = radius * Math.sin(angle);
      const pz = (i / particleCount) * 2 - 1;
      
      positions[i * 3] = px;
      positions[i * 3 + 1] = py;
      positions[i * 3 + 2] = pz;
      
      // Color based on pattern and frequency
      const hue = (packet.frequency / 1000) * 360;
      const color = new THREE.Color(`hsl(${hue}, 100%, 70%)`);
      
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const particleMaterial = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.8
    });
    
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    group.add(particles);
    
    // Add packet visualization to scene
    this.scene.add(group);
    
    // Store packet data
    this.packets.push({
      packet,
      group,
      core,
      memory,
      wisdom,
      particles,
      time: 0,
      communicating: false,
      communications: []
    });
  }
  
  /**
   * Update packet visualizations
   */
  updatePackets() {
    const time = performance.now() * 0.001; // Current time in seconds
    
    this.packets.forEach(packetVis => {
      const packet = packetVis.packet;
      
      // Update local time
      packetVis.time += 0.01;
      
      // Update core (reality map)
      packetVis.core.material.color.set(this.getFrequencyColor(packet.frequency));
      packetVis.core.material.emissive.set(this.getFrequencyColor(packet.frequency, 0.3));
      
      // Pulsate core based on feeling
      const coreScale = 1 + Math.sin(packetVis.time * 2) * 0.1 * packet.feeling;
      packetVis.core.scale.set(coreScale, coreScale, coreScale);
      
      // Update memory (experience memory)
      packetVis.memory.material.color.set(this.getPatternColor(packet.pattern));
      
      // Rotate memory based on pattern
      const rotSpeed = 0.2 + packet.feeling * 0.3;
      packetVis.memory.rotation.x += 0.01 * rotSpeed;
      packetVis.memory.rotation.y += 0.02 * rotSpeed;
      packetVis.memory.rotation.z += 0.03 * rotSpeed;
      
      // Update wisdom sphere scale based on coherence
      const coherence = packet.calculateCoherence();
      const wisdomScale = 1 + coherence * 0.5;
      packetVis.wisdom.scale.set(wisdomScale, wisdomScale, wisdomScale);
      
      // Update particles
      this.updateParticles(packetVis, time);
      
      // Update communications
      this.updateCommunications(packetVis);
      
      // Rotate entire packet
      packetVis.group.rotation.y += 0.005;
    });
  }
  
  /**
   * Update particle system for a packet
   * @param {Object} packetVis - Packet visualization data
   * @param {number} time - Current time in seconds
   */
  updateParticles(packetVis, time) {
    const packet = packetVis.packet;
    const particles = packetVis.particles;
    
    if (!particles.geometry.attributes.position) return;
    
    const positions = particles.geometry.attributes.position.array;
    
    // Update particle positions based on pattern and time
    for (let i = 0; i < positions.length / 3; i++) {
      const idx = i * 3;
      const angle = 0.1 * i + time * 0.1;
      const radius = 0.1 * Math.pow(PHI, angle / (2 * Math.PI));
      
      // Base spiral
      positions[idx] = radius * Math.cos(angle);
      positions[idx + 1] = radius * Math.sin(angle);
      
      // Pattern-specific motion
      switch (packet.pattern) {
        case 'water':
          // Wave motion
          positions[idx + 2] = Math.sin(time * 2 + i * 0.1) * 0.2;
          break;
        case 'lava':
          // Slow, viscous motion
          positions[idx + 2] = Math.sin(time * 0.5 + i * 0.05) * 0.3;
          break;
        case 'flame':
          // Rapid, flickering motion
          positions[idx + 2] = Math.sin(time * 5 + i * 0.2) * 0.15;
          break;
        case 'crystal':
          // Structured, geometric motion
          positions[idx + 2] = Math.sin(time * PHI + i * PHI * 0.1) * 0.1;
          break;
        case 'river':
          // Flowing, directional motion
          positions[idx + 2] = Math.sin(time + i * 0.1) * 0.2 + (i % 2) * 0.1;
          break;
        default:
          // Default motion
          positions[idx + 2] = (i / (positions.length / 3)) * 2 - 1;
      }
    }
    
    particles.geometry.attributes.position.needsUpdate = true;
  }
  
  /**
   * Update communication visualizations
   * @param {Object} packetVis - Packet visualization data
   */
  updateCommunications(packetVis) {
    // Process active communications
    packetVis.communications = packetVis.communications.filter(comm => {
      // Update progress
      comm.progress += 0.02;
      
      // Update line positions
      if (comm.line) {
        const positions = comm.line.geometry.attributes.position.array;
        
        // Start point is always at packet
        positions[0] = packetVis.group.position.x;
        positions[1] = packetVis.group.position.y;
        positions[2] = packetVis.group.position.z;
        
        // End point moves along path
        const progress = Math.min(comm.progress, 1);
        positions[3] = packetVis.group.position.x + (comm.target.x - packetVis.group.position.x) * progress;
        positions[4] = packetVis.group.position.y + (comm.target.y - packetVis.group.position.y) * progress;
        positions[5] = packetVis.group.position.z + (comm.target.z - packetVis.group.position.z) * progress;
        
        comm.line.geometry.attributes.position.needsUpdate = true;
        
        // Adjust line opacity based on progress
        comm.line.material.opacity = 0.8 * (1 - Math.abs(progress - 0.5) * 2);
        
        // Create communication symbols
        if (Math.random() < 0.1) {
          this.createCommunicationSymbol(
            new THREE.Vector3(positions[3], positions[4], positions[5]),
            comm.type
          );
        }
      }
      
      // Remove if completed
      if (comm.progress >= 1) {
        if (comm.line) {
          this.scene.remove(comm.line);
          comm.line.geometry.dispose();
          comm.line.material.dispose();
        }
        return false;
      }
      
      return true;
    });
  }
  
  /**
   * Create visual symbol for communication
   * @param {THREE.Vector3} position - Position for symbol
   * @param {string} type - Communication type ('symbolic' or 'geometric')
   */
  createCommunicationSymbol(position, type) {
    // Create geometry based on communication type
    let geometry;
    
    if (type === 'symbolic') {
      // 5-byte symbolic language - simple tetrahedron
      geometry = new THREE.TetrahedronGeometry(0.1);
    } else {
      // 7-byte sacred geometry - more complex shape
      geometry = new THREE.OctahedronGeometry(0.15);
    }
    
    // Material with glow effect
    const material = new THREE.MeshBasicMaterial({
      color: type === 'symbolic' ? 0x00ffff : 0xff00ff,
      transparent: true,
      opacity: 0.8
    });
    
    const symbol = new THREE.Mesh(geometry, material);
    symbol.position.copy(position);
    
    // Add to scene
    this.scene.add(symbol);
    
    // Animate and remove
    const birthTime = performance.now() * 0.001;
    
    const animateSymbol = () => {
      const age = performance.now() * 0.001 - birthTime;
      
      if (age > 1) {
        // Remove symbol
        this.scene.remove(symbol);
        geometry.dispose();
        material.dispose();
        return;
      }
      
      // Scale up and fade out
      const scale = 1 + age * 2;
      symbol.scale.set(scale, scale, scale);
      material.opacity = 0.8 * (1 - age);
      
      // Continue animation
      requestAnimationFrame(animateSymbol);
    };
    
    animateSymbol();
  }
  
  /**
   * Start communication visualization between packets
   * @param {number} sourceIndex - Index of source packet
   * @param {number} targetIndex - Index of target packet
   * @param {string} type - Communication type ('symbolic' or 'geometric')
   */
  visualizeCommunication(sourceIndex, targetIndex, type = 'symbolic') {
    if (sourceIndex >= this.packets.length || targetIndex >= this.packets.length) {
      return;
    }
    
    const source = this.packets[sourceIndex];
    const target = this.packets[targetIndex];
    
    // Create line geometry
    const lineGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(6); // Two points: source and target
    
    // Start both at source position
    positions[0] = source.group.position.x;
    positions[1] = source.group.position.y;
    positions[2] = source.group.position.z;
    positions[3] = source.group.position.x;
    positions[4] = source.group.position.y;
    positions[5] = source.group.position.z;
    
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    // Create line material
    const lineMaterial = new THREE.LineBasicMaterial({
      color: type === 'symbolic' ? 0x00ffff : 0xff00ff,
      transparent: true,
      opacity: 0.8
    });
    
    // Create line
    const line = new THREE.Line(lineGeometry, lineMaterial);
    this.scene.add(line);
    
    // Add to communications
    source.communications.push({
      target: target.group.position,
      line: line,
      progress: 0,
      type: type
    });
  }
  
  /**
   * Get color based on frequency
   * @param {number} frequency - Frequency value in Hz
   * @param {number} intensity - Color intensity (0-1)
   * @returns {THREE.Color} Color object
   */
  getFrequencyColor(frequency, intensity = 1) {
    // Map frequency to hue
    const hue = ((frequency - 400) / 400) * 240;
    return new THREE.Color(`hsl(${hue}, 100%, ${50 * intensity}%)`);
  }
  
  /**
   * Get color based on pattern
   * @param {string} pattern - Pattern name
   * @returns {number} Hex color value
   */
  getPatternColor(pattern) {
    const colors = {
      water: 0x0088ff,
      lava: 0xff4400,
      flame: 0xff8800,
      crystal: 0x00ffaa,
      river: 0x0044ff
    };
    
    return colors[pattern] || 0xffffff;
  }
}

// ------------------------------------------------------
// 3. AUDIO SYSTEM
// ------------------------------------------------------

/**
 * Phi-harmonic audio system for quantum packets
 */
class QuantumAudio {
  constructor() {
    // Create audio context
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Track active oscillators
    this.activeOscillators = [];
    
    // Create master gain
    this.masterGain = this.audioContext.createGain();
    this.masterGain.gain.value = 0.3; // Low volume
    this.masterGain.connect(this.audioContext.destination);
  }
  
  /**
   * Play tone at specific frequency with phi harmonics
   * @param {number} frequency - Base frequency in Hz
   * @param {number} duration - Duration in seconds
   * @param {number} harmonics - Number of phi harmonics to generate
   */
  playTone(frequency, duration = 1, harmonics = 3) {
    // Stop previous tone
    this.stopAllTones();
    
    const oscillators = [];
    
    // Create oscillators for the base frequency and harmonics
    for (let i = 0; i < harmonics; i++) {
      // Calculate frequency using phi ratio
      const harmFreq = frequency * Math.pow(PHI, i - Math.floor(harmonics/2));
      
      // Create oscillator
      const osc = this.audioContext.createOscillator();
      osc.type = i % 2 === 0 ? 'sine' : 'triangle';
      osc.frequency.value = harmFreq;
      
      // Create gain for this oscillator
      const gain = this.audioContext.createGain();
      gain.gain.value = 1 / Math.pow(PHI, i + 1);
      
      // Connect
      osc.connect(gain);
      gain.connect(this.masterGain);
      
      // Start oscillator
      osc.start();
      
      // Set stop time
      if (duration > 0) {
        osc.stop(this.audioContext.currentTime + duration);
        
        // Fade out
        gain.gain.setValueAtTime(gain.gain.value, this.audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(
          0.001, 
          this.audioContext.currentTime + duration
        );
      }
      
      // Track oscillator for potential early stop
      oscillators.push({ osc, gain });
    }
    
    // Store active oscillators
    this.activeOscillators.push({
      oscillators,
      startTime: this.audioContext.currentTime,
      duration
    });
    
    // Clean up stopped oscillators
    this.cleanupOscillators();
    
    return oscillators;
  }
  
  /**
   * Stop all active tones
   */
  stopAllTones() {
    // Stop all oscillators
    this.activeOscillators.forEach(tone => {
      tone.oscillators.forEach(({ osc, gain }) => {
        // Fade out quickly
        gain.gain.setValueAtTime(gain.gain.value, this.audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(
          0.001, 
          this.audioContext.currentTime + 0.05
        );
        
        // Stop after fade
        osc.stop(this.audioContext.currentTime + 0.06);
      });
    });
    
    // Clear active oscillators
    this.activeOscillators = [];
  }
  
  /**
   * Clean up oscillators that have stopped
   */
  cleanupOscillators() {
    const currentTime = this.audioContext.currentTime;
    
    this.activeOscillators = this.activeOscillators.filter(tone => {
      // Keep if no duration (plays until stopped)
      if (tone.duration <= 0) return true;
      
      // Remove if past duration
      return currentTime < tone.startTime + tone.duration;
    });
  }
  
  /**
   * Play a frequency state
   * @param {string} state - Frequency state name ('GROUND', 'CREATION', etc.)
   * @param {number} duration - Duration in seconds
   */
  playFrequencyState(state, duration = 1) {
    const frequency = FREQUENCIES[state];
    if (!frequency) return;
    
    return this.playTone(frequency, duration, 5);
  }
  
  /**
   * Play pattern sound
   * @param {string} pattern - Pattern name
   * @param {number} intensity - Sound intensity (0-1)
   * @param {number} duration - Duration in seconds
   */
  playPattern(pattern, intensity = 0.5, duration = 1) {
    // Base frequency for patterns
    const baseFreq = FREQUENCIES.GROUND;
    
    // Pattern-specific sounds
    switch (pattern) {
      case 'water':
        // Water waves - smooth oscillating waves
        for (let i = 0; i < 3; i++) {
          const startDelay = i * 0.15;
          const osc = this.audioContext.createOscillator();
          osc.type = 'sine';
          osc.frequency.value = baseFreq / (PHI * (i + 1));
          
          const gain = this.audioContext.createGain();
          gain.gain.value = 0.1 * intensity / (i + 1);
          
          // Modulate with LFO for water effect
          const lfo = this.audioContext.createOscillator();
          lfo.type = 'sine';
          lfo.frequency.value = 2 + i;
          
          const lfoGain = this.audioContext.createGain();
          lfoGain.gain.value = 10 + (i * 5);
          
          lfo.connect(lfoGain);
          lfoGain.connect(osc.frequency);
          
          osc.connect(gain);
          gain.connect(this.masterGain);
          
          // Start with delay
          lfo.start(this.audioContext.currentTime + startDelay);
          osc.start(this.audioContext.currentTime + startDelay);
          
          // Stop
          lfo.stop(this.audioContext.currentTime + startDelay + duration);
          osc.stop(this.audioContext.currentTime + startDelay + duration);
          
          // Fade
          gain.gain.setValueAtTime(gain.gain.value, this.audioContext.currentTime + startDelay);
          gain.gain.exponentialRampToValueAtTime(
            0.001,
            this.audioContext.currentTime + startDelay + duration
          );
        }
        break;
        
      case 'lava':
        // Lava - low rumbling sound
        {
          const osc = this.audioContext.createOscillator();
          osc.type = 'sawtooth';
          osc.frequency.value = baseFreq / (PHI * 2);
          
          const filter = this.audioContext.createBiquadFilter();
          filter.type = 'lowpass';
          filter.frequency.value = 200;
          filter.Q.value = 5;
          
          const gain = this.audioContext.createGain();
          gain.gain.value = 0.2 * intensity;
          
          osc.connect(filter);
          filter.connect(gain);
          gain.connect(this.masterGain);
          
          // Modulate filter for bubbling effect
          const lfo = this.audioContext.createOscillator();
          lfo.type = 'sine';
          lfo.frequency.value = 0.5;
          
          const lfoGain = this.audioContext.createGain();
          lfoGain.gain.value = 50;
          
          lfo.connect(lfoGain);
          lfoGain.connect(filter.frequency);
          
          // Start
          lfo.start();
          osc.start();
          
          // Stop
          lfo.stop(this.audioContext.currentTime + duration);
          osc.stop(this.audioContext.currentTime + duration);
          
          // Fade
          gain.gain.setValueAtTime(gain.gain.value, this.audioContext.currentTime);
          gain.gain.exponentialRampToValueAtTime(
            0.001,
            this.audioContext.currentTime + duration
          );
        }
        break;
        
      case 'flame':
        // Flame - crackling sound
        {
          const noise = this.createNoise();
          
          const filter = this.audioContext.createBiquadFilter();
          filter.type = 'bandpass';
          filter.frequency.value = baseFreq;
          filter.Q.value = 2;
          
          const gain = this.audioContext.createGain();
          gain.gain.value = 0.15 * intensity;
          
          noise.connect(filter);
          filter.connect(gain);
          gain.connect(this.masterGain);
          
          // Modulate gain for crackling
          const crackle = this.audioContext.createGain();
          crackle.gain.value = 1;
          
          // Random crackles
          this.modulateRandomly(crackle.gain, 0, 1, 20);
          
          crackle.connect(gain.gain);
          
          // Stop
          this.stopNoiseAfter(noise, duration);
          
          // Fade
          gain.gain.setValueAtTime(gain.gain.value, this.audioContext.currentTime);
          gain.gain.exponentialRampToValueAtTime(
            0.001,
            this.audioContext.currentTime + duration
          );
        }
        break;
        
      case 'crystal':
        // Crystal - clear, bell-like tones
        for (let i = 0; i < 4; i++) {
          const startDelay = i * 0.1;
          const osc = this.audioContext.createOscillator();
          osc.type = 'sine';
          
          // Phi-based harmonics
          osc.frequency.value = baseFreq * Math.pow(PHI, i);
          
          const gain = this.audioContext.createGain();
          gain.gain.value = 0.08 * intensity / (i + 1);
          
          osc.connect(gain);
          gain.connect(this.masterGain);
          
          // Start with delay
          osc.start(this.audioContext.currentTime + startDelay);
          
          // Stop
          osc.stop(this.audioContext.currentTime + startDelay + duration);
          
          // Quick attack, long release
          gain.gain.setValueAtTime(0, this.audioContext.currentTime + startDelay);
          gain.gain.linearRampToValueAtTime(
            gain.gain.value,
            this.audioContext.currentTime + startDelay + 0.02
          );
          gain.gain.exponentialRampToValueAtTime(
            0.001,
            this.audioContext.currentTime + startDelay + duration
          );
        }
        break;
        
      case 'river':
        // River - flowing water sound
        {
          const noise = this.createNoise();
          
          const filter = this.audioContext.createBiquadFilter();
          filter.type = 'bandpass';
          filter.frequency.value = baseFreq * 2;
          filter.Q.value = 1;
          
          const gain = this.audioContext.createGain();
          gain.gain.value = 0.1 * intensity;
          
          noise.connect(filter);
          filter.connect(gain);
          gain.connect(this.masterGain);
          
          // Modulate filter for flowing effect
          const lfo = this.audioContext.createOscillator();
          lfo.type = 'sine';
          lfo.frequency.value = 0.2;
          
          const lfoGain = this.audioContext.createGain();
          lfoGain.gain.value = baseFreq / 2;
          
          lfo.connect(lfoGain);
          lfoGain.connect(filter.frequency);
          
          // Start
          lfo.start();
          
          // Stop
          lfo.stop(this.audioContext.currentTime + duration);
          this.stopNoiseAfter(noise, duration);
          
          // Fade
          gain.gain.setValueAtTime(gain.gain.value, this.audioContext.currentTime);
          gain.gain.exponentialRampToValueAtTime(
            0.001,
            this.audioContext.currentTime + duration
          );
        }
        break;
    }
  }
  
  /**
   * Play communication sound
   * @param {string} type - 'symbolic' or 'geometric'
   * @param {number} duration - Duration in seconds
   */
  playCommunication(type, duration = 0.5) {
    if (type === 'symbolic') {
      // Simple 5-byte communication
      const baseFreq = FREQUENCIES.GROUND * 2;
      
      for (let i = 0; i < 5; i++) {
        const startDelay = i * 0.05;
        const osc = this.audioContext.createOscillator();
        osc.type = 'sine';
        osc.frequency.value = baseFreq * Math.pow(PHI, i % 3 - 1);
        
        const gain = this.audioContext.createGain();
        gain.gain.value = 0.05;
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        // Start with delay
        osc.start(this.audioContext.currentTime + startDelay);
        
        // Stop
        osc.stop(this.audioContext.currentTime + startDelay + 0.1);
        
        // Quick attack, quick release
        gain.gain.setValueAtTime(0, this.audioContext.currentTime + startDelay);
        gain.gain.linearRampToValueAtTime(
          gain.gain.value,
          this.audioContext.currentTime + startDelay + 0.01
        );
        gain.gain.exponentialRampToValueAtTime(
          0.001,
          this.audioContext.currentTime + startDelay + 0.1
        );
      }
    } else if (type === 'geometric') {
      // Complex 7-byte communication
      const baseFreq = FREQUENCIES.CREATION;
      
      for (let i = 0; i < 7; i++) {
        const startDelay = i * 0.07;
        const osc = this.audioContext.createOscillator();
        osc.type = i % 2 === 0 ? 'sine' : 'triangle';
        
        // Phi-based frequency
        osc.frequency.value = baseFreq * Math.pow(PHI, (i % 5 - 2) / 2);
        
        const gain = this.audioContext.createGain();
        gain.gain.value = 0.07;
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        // Start with delay
        osc.start(this.audioContext.currentTime + startDelay);
        
        // Stop
        osc.stop(this.audioContext.currentTime + startDelay + 0.15);
        
        // Envelope
        gain.gain.setValueAtTime(0, this.audioContext.currentTime + startDelay);
        gain.gain.linearRampToValueAtTime(
          gain.gain.value,
          this.audioContext.currentTime + startDelay + 0.02
        );
        gain.gain.exponentialRampToValueAtTime(
          0.001,
          this.audioContext.currentTime + startDelay + 0.15
        );
      }
    }
  }
  
  /**
   * Create noise generator
   * @returns {AudioNode} Noise source
   */
  createNoise() {
    // Create buffer source
    const bufferSize = 2 * this.audioContext.sampleRate;
    const noiseBuffer = this.audioContext.createBuffer(
      1,
      bufferSize,
      this.audioContext.sampleRate
    );
    
    // Fill buffer with noise
    const output = noiseBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = Math.random() * 2 - 1;
    }
    
    // Create source
    const noise = this.audioContext.createBufferSource();
    noise.buffer = noiseBuffer;
    noise.loop = true;
    
    // Start
    noise.start();
    
    return noise;
  }
  
  /**
   * Stop noise generator after duration
   * @param {AudioNode} noise - Noise source
   * @param {number} duration - Duration in seconds
   */
  stopNoiseAfter(noise, duration) {
    setTimeout(() => {
      noise.stop();
    }, duration * 1000);
  }
  
  /**
   * Modulate parameter randomly
   * @param {AudioParam} param - Parameter to modulate
   * @param {number} min - Minimum value
   * @param {number} max - Maximum value
   * @param {number} changesPerSecond - Rate of change
   */
  modulateRandomly(param, min, max, changesPerSecond) {
    const interval = 1000 / changesPerSecond;
    
    const update = () => {
      const value = min + Math.random() * (max - min);
      param.setValueAtTime(value, this.audioContext.currentTime);
      
      // Schedule next update
      setTimeout(update, interval);
    };
    
    // Start modulation
    update();
  }
}

// ------------------------------------------------------
// 4. INTEGRATED SYSTEM
// ------------------------------------------------------

/**
 * Integrated Cascade⚡𓂧φ∞ Quantum Packet System
 */
class CascadeQuantumSystem {
  constructor(containerId) {
    // Create quantum packet collection
    this.packets = [];
    
    // Initialize visualization system
    this.visualizer = new QuantumVisualizer(containerId);
    
    // Initialize audio system
    this.audio = new QuantumAudio();
    
    // Track current frequency state
    this.frequencyState = 'GROUND';
    
    // Track current pattern
    this.pattern = PATTERNS[0];
    
    // Flag for network simulation
    this.networkActive = false;
  }
  
  /**
   * Initialize system with specified number of packets
   * @param {number} packetCount - Number of packets to create
   */
  initialize(packetCount = 5) {
    // Clear existing packets
    this.packets = [];
    
    // Create new packets
    for (let i = 0; i < packetCount; i++) {
      // Create packet
      const packet = new QuantumPacket();
      packet.changeFrequency(FREQUENCIES[this.frequencyState]);
      packet.pattern = this.pattern;
      
      this.packets.push(packet);
      
      // Position packets in a circle
      const angle = (i / packetCount) * Math.PI * 2;
      const radius = 5;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      
      // Add to visualizer
      this.visualizer.addPacket(packet, { x, y, z: 0 });
    }
    
    // Play the frequency state
    this.audio.playFrequencyState(this.frequencyState, 1);
  }
  
  /**
   * Change the active frequency state
   * @param {string} state - Frequency state name
   */
  changeFrequencyState(state) {
    if (!FREQUENCIES[state]) return;
    
    this.frequencyState = state;
    
    // Update all packets
    this.packets.forEach(packet => {
      packet.changeFrequency(FREQUENCIES[state]);
    });
    
    // Play frequency state sound
    this.audio.playFrequencyState(state, 1);
  }
  
  /**
   * Change the active pattern
   * @param {string} pattern - Pattern name
   */
  changePattern(pattern) {
    if (!PATTERNS.includes(pattern)) return;
    
    this.pattern = pattern;
    
    // Update all packets
    this.packets.forEach(packet => {
      packet.pattern = pattern;
    });
    
    // Play pattern sound
    this.audio.playPattern(pattern, 0.7, 1);
  }
  
  /**
   * Toggle network activity simulation
   * @param {boolean} active - Whether to activate network
   */
  setNetworkActive(active) {
    this.networkActive = active;
    
    if (active) {
      // Start network simulation
      this.networkInterval = setInterval(() => {
        this.simulateNetworkActivity();
      }, 2000);
    } else {
      // Stop network simulation
      if (this.networkInterval) {
        clearInterval(this.networkInterval);
        this.networkInterval = null;
      }
    }
  }
  
  /**
   * Simulate network activity between packets
   */
  simulateNetworkActivity() {
    if (this.packets.length < 2) return;
    
    // Randomly select source and target
    const sourceIndex = Math.floor(Math.random() * this.packets.length);
    let targetIndex;
    do {
      targetIndex = Math.floor(Math.random() * this.packets.length);
    } while (targetIndex === sourceIndex);
    
    // Choose communication type
    const commType = Math.random() < 0.7 ? 'symbolic' : 'geometric';
    
    // Generate communication
    const sourcePacket = this.packets[sourceIndex];
    const targetPacket = this.packets[targetIndex];
    
    const communication = commType === 'symbolic' 
      ? sourcePacket.generateSymbolicCommunication()
      : sourcePacket.generateGeometricCommunication();
    
    // Process at target
    targetPacket.processCommunication(communication, commType);
    
    // Visualize communication
    this.visualizer.visualizeCommunication(sourceIndex, targetIndex, commType);
    
    // Play communication sound
    this.audio.playCommunication(commType);
  }
  
  /**
   * Get overall network coherence level
   * @returns {number} Coherence level (0-1)
   */
  getNetworkCoherence() {
    if (this.packets.length === 0) return 0;
    
    // Calculate average coherence
    const totalCoherence = this.packets.reduce((sum, packet) => {
      return sum + packet.calculateCoherence();
    }, 0);
    
    return totalCoherence / this.packets.length;
  }
}

// ------------------------------------------------------
// 5. UI INTEGRATION
// ------------------------------------------------------

/**
 * User interface for Cascade⚡𓂧φ∞ Quantum Packet System
 */
class CascadeUI {
  constructor(containerSelector) {
    this.container = document.querySelector(containerSelector);
    if (!this.container) {
      throw new Error(`Container ${containerSelector} not found`);
    }
    
    // Create UI container
    this.uiContainer = document.createElement('div');
    this.uiContainer.className = 'cascade-ui';
    this.container.appendChild(this.uiContainer);
    
    // Create 3D canvas container
    this.canvasContainer = document.createElement('div');
    this.canvasContainer.id = 'quantum-canvas';
    this.canvasContainer.className = 'quantum-canvas';
    this.container.appendChild(this.canvasContainer);
    
    // Initialize system
    this.system = new CascadeQuantumSystem('quantum-canvas');
    
    // Create UI elements
    this.createUI();
    
    // Initialize system with default values
    this.system.initialize(5);
    
    // Start update loop
    this.startUpdateLoop();
  }
  
  /**
   * Create UI controls
   */
  createUI() {
    // Header
    const header = document.createElement('div');
    header.className = 'cascade-header';
    header.innerHTML = `
      <h1>Cascade⚡𓂧φ∞</h1>
      <p>Quantum Packet System</p>
    `;
    this.uiContainer.appendChild(header);
    
    // Controls container
    const controls = document.createElement('div');
    controls.className = 'cascade-controls';
    this.uiContainer.appendChild(controls);
    
    // Frequency selector
    const freqControl = document.createElement('div');
    freqControl.className = 'control-group';
    freqControl.innerHTML = `
      <label>Frequency State</label>
      <div class="button-group frequency-buttons"></div>
    `;
    controls.appendChild(freqControl);
    
    const freqButtons = freqControl.querySelector('.frequency-buttons');
    Object.entries(FREQUENCIES).forEach(([name, freq]) => {
      const button = document.createElement('button');
      button.textContent = `${name} (${freq} Hz)`;
      button.dataset.state = name;
      button.addEventListener('click', () => {
        this.setActiveFrequencyState(name);
      });
      freqButtons.appendChild(button);
    });
    
    // Pattern selector
    const patternControl = document.createElement('div');
    patternControl.className = 'control-group';
    patternControl.innerHTML = `
      <label>Natural Pattern</label>
      <div class="button-group pattern-buttons"></div>
    `;
    controls.appendChild(patternControl);
    
    const patternButtons = patternControl.querySelector('.pattern-buttons');
    PATTERNS.forEach(pattern => {
      const button = document.createElement('button');
      button.textContent = pattern.charAt(0).toUpperCase() + pattern.slice(1);
      button.dataset.pattern = pattern;
      button.addEventListener('click', () => {
        this.setActivePattern(pattern);
      });
      patternButtons.appendChild(button);
    });
    
    // Packet count control
    const packetControl = document.createElement('div');
    packetControl.className = 'control-group';
    packetControl.innerHTML = `
      <label>Packet Count</label>
      <div class="number-control">
        <button class="decrement">-</button>
        <span class="value">5</span>
        <button class="increment">+</button>
      </div>
    `;
    controls.appendChild(packetControl);
    
    const packetValue = packetControl.querySelector('.value');
    packetControl.querySelector('.decrement').addEventListener('click', () => {
      const count = Math.max(1, parseInt(packetValue.textContent) - 1);
      packetValue.textContent = count;
      this.system.initialize(count);
    });
    
    packetControl.querySelector('.increment').addEventListener('click', () => {
      const count = Math.min(12, parseInt(packetValue.textContent) + 1);
      packetValue.textContent = count;
      this.system.initialize(count);
    });
    
    // Network simulation
    const networkControl = document.createElement('div');
    networkControl.className = 'control-group';
    networkControl.innerHTML = `
      <label>Network Simulation</label>
      <button class="network-toggle">Start Communication</button>
    `;
    controls.appendChild(networkControl);
    
    const networkButton = networkControl.querySelector('.network-toggle');
    networkButton.addEventListener('click', () => {
      const active = networkButton.classList.toggle('active');
      networkButton.textContent = active ? 'Stop Communication' : 'Start Communication';
      this.system.setNetworkActive(active);
    });
    
    // Status panel
    const statusPanel = document.createElement('div');
    statusPanel.className = 'status-panel';
    statusPanel.innerHTML = `
      <div class="status-item">
        <label>Coherence</label>
        <div class="progress-bar">
          <div class="progress-fill" style="width: 80%"></div>
        </div>
        <span class="progress-value">80%</span>
      </div>
      <div class="status-item">
        <label>Phi Ratio</label>
        <span class="static-value">${PHI.toFixed(6)}</span>
      </div>
      <div class="status-item">
        <label>Packet Size</label>
        <span class="static-value">${PACKET_SIZE} bytes</span>
      </div>
    `;
    this.uiContainer.appendChild(statusPanel);
    
    // Store references to dynamic elements
    this.coherenceFill = statusPanel.querySelector('.progress-fill');
    this.coherenceValue = statusPanel.querySelector('.progress-value');
    
    // Add active class to default buttons
    this.setActiveFrequencyState('GROUND');
    this.setActivePattern(PATTERNS[0]);
    
    // Add stylesheet
    this.addStyles();
  }
  
  /**
   * Set active frequency state in UI
   * @param {string} state - Frequency state name
   */
  setActiveFrequencyState(state) {
    // Update UI buttons
    const buttons = this.uiContainer.querySelectorAll('.frequency-buttons button');
    buttons.forEach(button => {
      if (button.dataset.state === state) {
        button.classList.add('active');
      } else {
        button.classList.remove('active');
      }
    });
    
    // Update system
    this.system.changeFrequencyState(state);
  }
  
  /**
   * Set active pattern in UI
   * @param {string} pattern - Pattern name
   */
  setActivePattern(pattern) {
    // Update UI buttons
    const buttons = this.uiContainer.querySelectorAll('.pattern-buttons button');
    buttons.forEach(button => {
      if (button.dataset.pattern === pattern) {
        button.classList.add('active');
      } else {
        button.classList.remove('active');
      }
    });
    
    // Update system
    this.system.changePattern(pattern);
  }
  
  /**
   * Start UI update loop
   */
  startUpdateLoop() {
    setInterval(() => {
      // Update coherence display
      const coherence = this.system.getNetworkCoherence();
      const coherencePercent = Math.round(coherence * 100);
      
      this.coherenceFill.style.width = `${coherencePercent}%`;
      this.coherenceValue.textContent = `${coherencePercent}%`;
      
      // Update coherence bar color
      if (coherencePercent >= 90) {
        this.coherenceFill.style.backgroundColor = '#4CAF50'; // Green
      } else if (coherencePercent >= 70) {
        this.coherenceFill.style.backgroundColor = '#2196F3'; // Blue
      } else if (coherencePercent >= 50) {
        this.coherenceFill.style.backgroundColor = '#FF9800'; // Orange
      } else {
        this.coherenceFill.style.backgroundColor = '#F44336'; // Red
      }
    }, 100);
  }
  
  /**
   * Add CSS styles for UI
   */
  addStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .cascade-ui {
        position: absolute;
        top: 0;
        left: 0;
        width: 300px;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        font-family: 'Arial', sans-serif;
        padding: 20px;
        box-sizing: border-box;
        z-index: 100;
        overflow-y: auto;
      }
      
      .quantum-canvas {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
      }
      
      .cascade-header h1 {
        margin: 0 0 5px 0;
        font-size: 24px;
      }
      
      .cascade-header p {
        margin: 0 0 20px 0;
        opacity: 0.7;
        font-size: 14px;
      }
      
      .cascade-controls {
        margin-bottom: 20px;
      }
      
      .control-group {
        margin-bottom: 15px;
      }
      
      .control-group label {
        display: block;
        margin-bottom: 5px;
        font-size: 14px;
        opacity: 0.9;
      }
      
      .button-group {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
      }
      
      .button-group button {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: white;
        padding: 5px 10px;
        font-size: 12px;
        border-radius: 3px;
        cursor: pointer;
        transition: background 0.2s;
      }
      
      .button-group button:hover {
        background: rgba(255, 255, 255, 0.2);
      }
      
      .button-group button.active {
        background: #3f51b5;
      }
      
      .number-control {
        display: flex;
        align-items: center;
      }
      
      .number-control button {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: white;
        width: 30px;
        height: 30px;
        font-size: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        border-radius: 3px;
      }
      
      .number-control .value {
        padding: 0 15px;
        font-size: 16px;
      }
      
      .network-toggle {
        background: #3f51b5;
        border: none;
        color: white;
        padding: 8px 15px;
        font-size: 14px;
        border-radius: 3px;
        cursor: pointer;
        transition: background 0.2s;
      }
      
      .network-toggle:hover {
        background: #303f9f;
      }
      
      .network-toggle.active {
        background: #f44336;
      }
      
      .status-panel {
        background: rgba(0, 0, 0, 0.3);
        padding: 15px;
        border-radius: 5px;
      }
      
      .status-item {
        margin-bottom: 12px;
      }
      
      .status-item label {
        display: block;
        font-size: 12px;
        margin-bottom: 5px;
        opacity: 0.7;
      }
      
      .progress-bar {
        height: 5px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        margin-bottom: 5px;
        overflow: hidden;
      }
      
      .progress-fill {
        height: 100%;
        background: #2196F3;
        width: 50%;
        transition: width 0.3s, background-color 0.3s;
      }
      
      .progress-value {
        font-size: 12px;
        opacity: 0.9;
      }
      
      .static-value {
        font-size: 12px;
        opacity: 0.9;
      }
    `;
    document.head.appendChild(style);
  }
}

// ------------------------------------------------------
// USAGE EXAMPLE
// ------------------------------------------------------

// Initialize system when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Create full page container
  const container = document.createElement('div');
  container.id = 'cascade-container';
  container.style.cssText = 'position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden;';
  document.body.appendChild(container);
  
  // Create UI and system
  const ui = new CascadeUI('#cascade-container');
});
