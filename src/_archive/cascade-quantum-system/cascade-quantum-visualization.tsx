import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const PHI = 1.618033988749895;
const PACKET_SIZE = 432;

// Frequency constants
const FREQUENCIES = {
  GROUND: 432,  // Foundation/stability
  CREATION: 528, // Creation/pattern formation
  HEART: 594,   // Emotional resonance
  UNITY: 768,   // Perfect integration
};

// Natural patterns
const PATTERNS = ["water", "lava", "flame", "crystal", "river"];

const CascadeQuantumVisualizer = () => {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const frameIdRef = useRef(null);
  const timeRef = useRef(0);
  const packetsRef = useRef([]);
  
  const [activeFrequency, setActiveFrequency] = useState(FREQUENCIES.GROUND);
  const [activePattern, setActivePattern] = useState(PATTERNS[0]);
  const [coherenceLevel, setCoherenceLevel] = useState(0.85);
  const [packetCount, setPacketCount] = useState(5);
  const [networkActive, setNetworkActive] = useState(false);
  const [selectedPacket, setSelectedPacket] = useState(null);
  
  // Initialize the scene
  useEffect(() => {
    if (!mountRef.current) return;
    
    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111122);
    sceneRef.current = scene;
    
    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 15;
    cameraRef.current = camera;
    
    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x333333);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);
    
    // Initialize packets
    initializePackets();
    
    // Handle window resize
    const handleResize = () => {
      if (rendererRef.current && cameraRef.current) {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        rendererRef.current.setSize(width, height);
        cameraRef.current.aspect = width / height;
        cameraRef.current.updateProjectionMatrix();
      }
    };
    
    window.addEventListener('resize', handleResize);
    
    // Animation loop
    const animate = () => {
      timeRef.current += 0.01;
      
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      
      // Animate packets
      animatePackets();
      
      // Simulate network activity if enabled
      if (networkActive) {
        simulateNetworkActivity();
      }
      
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
      
      frameIdRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameIdRef.current);
      
      if (mountRef.current && rendererRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      
      // Dispose resources
      if (sceneRef.current) {
        disposeScene(sceneRef.current);
      }
    };
  }, []);
  
  // Update when packet count changes
  useEffect(() => {
    initializePackets();
  }, [packetCount]);
  
  // Initialize quantum packets
  const initializePackets = () => {
    if (!sceneRef.current) return;
    
    // Remove existing packets
    packetsRef.current.forEach(packet => {
      if (packet.group && sceneRef.current) {
        sceneRef.current.remove(packet.group);
      }
    });
    
    packetsRef.current = [];
    
    // Create new packets
    for (let i = 0; i < packetCount; i++) {
      createPacket(i);
    }
  };
  
  // Create a single quantum packet
  const createPacket = (index) => {
    if (!sceneRef.current) return;
    
    // Create group for the packet
    const group = new THREE.Group();
    
    // Base position
    const angle = (index / packetCount) * Math.PI * 2;
    const radius = 5;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    const z = 0;
    
    group.position.set(x, y, z);
    
    // Create packet core (reality map)
    const coreGeometry = new THREE.IcosahedronGeometry(0.5, 2);
    const coreMaterial = new THREE.MeshPhongMaterial({
      color: getFrequencyColor(activeFrequency),
      emissive: getFrequencyColor(activeFrequency, 0.3),
      shininess: 30,
      transparent: true,
      opacity: 0.9
    });
    
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    group.add(core);
    
    // Create experience memory layer
    const memoryGeometry = new THREE.TorusKnotGeometry(0.8, 0.2, 64, 8, 2, 3);
    const memoryMaterial = new THREE.MeshPhongMaterial({
      color: getPatternColor(activePattern),
      transparent: true,
      opacity: 0.7,
      wireframe: false
    });
    
    const memory = new THREE.Mesh(memoryGeometry, memoryMaterial);
    group.add(memory);
    
    // Create wisdom accumulator field
    const wisdomGeometry = new THREE.SphereGeometry(1.2, 16, 16);
    const wisdomMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.2,
      wireframe: true
    });
    
    const wisdom = new THREE.Mesh(wisdomGeometry, wisdomMaterial);
    group.add(wisdom);
    
    // Create golden spiral particle system for patterns
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
      const hue = (activeFrequency / 1000) * 360;
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
    
    // Add communication lines
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.2
    });
    
    const lineGeometries = [];
    
    // Add packet to scene
    sceneRef.current.add(group);
    
    // Create packet data structure
    const packet = {
      id: index,
      group,
      core,
      memory,
      wisdom,
      particles,
      lines: [],
      lineGeometries,
      frequency: activeFrequency,
      pattern: activePattern,
      feeling: Math.random() * 0.5 + 0.5,
      experience: PATTERNS.map(() => Math.random() * 0.3),
      communicating: false,
      targetPacket: null,
      communicationType: null,
      communicationProgress: 0
    };
    
    // Make packet interactive
    core.userData.packetId = index;
    memory.userData.packetId = index;
    wisdom.userData.packetId = index;
    
    // Store packet
    packetsRef.current.push(packet);
    
    return packet;
  };
  
  // Animate packets
  const animatePackets = () => {
    const time = timeRef.current;
    
    packetsRef.current.forEach((packet, i) => {
      if (!packet.group) return;
      
      // Base rotation
      packet.group.rotation.y = time * 0.2 + i * 0.1;
      
      // Core pulsation based on frequency
      const coreScale = 1 + Math.sin(time * packet.frequency / 432) * 0.1;
      packet.core.scale.set(coreScale, coreScale, coreScale);
      
      // Memory rotation based on experience
      packet.memory.rotation.x = time * 0.3;
      packet.memory.rotation.y = time * 0.2;
      packet.memory.rotation.z = time * 0.1;
      
      // Wisdom field pulsation based on feeling
      const wisdomScale = 1 + Math.sin(time * packet.feeling * 2) * 0.1;
      packet.wisdom.scale.set(wisdomScale, wisdomScale, wisdomScale);
      
      // Update particles
      const positions = packet.particles.geometry.attributes.position.array;
      
      for (let j = 0; j < positions.length / 3; j++) {
        const idx = j * 3;
        const angle = 0.1 * j + time * 0.1;
        const radius = 0.1 * Math.pow(PHI, angle / (2 * Math.PI));
        
        positions[idx] = radius * Math.cos(angle);
        positions[idx + 1] = radius * Math.sin(angle);
        // Z position oscillates based on pattern
        positions[idx + 2] = (j / (positions.length / 3)) * 2 - 1 + 
                           Math.sin(time * packet.feeling + j * 0.1) * 0.2;
      }
      
      packet.particles.geometry.attributes.position.needsUpdate = true;
      
      // Handle communication visualization
      if (packet.communicating) {
        updateCommunicationVisualization(packet);
      }
    });
    
    // Update field coherence
    const totalFeeling = packetsRef.current.reduce((sum, p) => sum + p.feeling, 0);
    const networkCoherence = totalFeeling / packetsRef.current.length;
    setCoherenceLevel(networkCoherence);
  };
  
  // Simulate network activity between packets
  const simulateNetworkActivity = () => {
    // Randomly initiate communication between packets
    if (Math.random() < 0.01 && packetsRef.current.length > 1) {
      // Select source packet
      const sourceIndex = Math.floor(Math.random() * packetsRef.current.length);
      
      // Select target packet (different from source)
      let targetIndex;
      do {
        targetIndex = Math.floor(Math.random() * packetsRef.current.length);
      } while (targetIndex === sourceIndex);
      
      const sourcePacket = packetsRef.current[sourceIndex];
      const targetPacket = packetsRef.current[targetIndex];
      
      // Don't start new communication if already communicating
      if (sourcePacket.communicating || targetPacket.communicating) {
        return;
      }
      
      // Choose communication type
      const communicationType = Math.random() < 0.5 ? 'symbolic' : 'geometric';
      
      // Initiate communication
      initiatePacketCommunication(sourcePacket, targetPacket, communicationType);
    }
  };
  
  // Initiate communication between packets
  const initiatePacketCommunication = (sourcePacket, targetPacket, type) => {
    sourcePacket.communicating = true;
    sourcePacket.targetPacket = targetPacket;
    sourcePacket.communicationType = type;
    sourcePacket.communicationProgress = 0;
    
    // Create communication line
    const sourcePosition = new THREE.Vector3();
    sourcePacket.group.getWorldPosition(sourcePosition);
    
    const targetPosition = new THREE.Vector3();
    targetPacket.group.getWorldPosition(targetPosition);
    
    // Create line geometry
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([
      sourcePosition,
      sourcePosition.clone() // Start with both points at source
    ]);
    
    const lineMaterial = new THREE.LineBasicMaterial({
      color: type === 'symbolic' ? 0x00ffff : 0xff00ff,
      transparent: true,
      opacity: 0.8
    });
    
    const line = new THREE.Line(lineGeometry, lineMaterial);
    sceneRef.current.add(line);
    
    sourcePacket.lines.push({
      object: line,
      target: targetPosition,
      type
    });
  };
  
  // Update communication visualization
  const updateCommunicationVisualization = (packet) => {
    if (!packet.communicating || !packet.lines.length) return;
    
    packet.communicationProgress += 0.01;
    
    // Update each communication line
    packet.lines.forEach(line => {
      const progress = Math.min(packet.communicationProgress, 1);
      
      const sourcePosition = new THREE.Vector3();
      packet.group.getWorldPosition(sourcePosition);
      
      const positions = line.object.geometry.attributes.position.array;
      
      // Update start point to current position
      positions[0] = sourcePosition.x;
      positions[1] = sourcePosition.y;
      positions[2] = sourcePosition.z;
      
      // Update end point based on progress
      positions[3] = sourcePosition.x + (line.target.x - sourcePosition.x) * progress;
      positions[4] = sourcePosition.y + (line.target.y - sourcePosition.y) * progress;
      positions[5] = sourcePosition.z + (line.target.z - sourcePosition.z) * progress;
      
      line.object.geometry.attributes.position.needsUpdate = true;
      
      // Adjust line opacity based on progress
      line.object.material.opacity = 0.8 * (1 - Math.abs(progress - 0.5) * 2);
      
      // Create packet data symbols at the communication point
      if (Math.random() < 0.1 && sceneRef.current) {
        const symbolPosition = new THREE.Vector3(
          positions[3],
          positions[4],
          positions[5]
        );
        
        createCommunicationSymbol(symbolPosition, line.type);
      }
    });
    
    // Check if communication is complete
    if (packet.communicationProgress >= 1) {
      completeCommunication(packet);
    }
  };
  
  // Create visual symbol for communication
  const createCommunicationSymbol = (position, type) => {
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
    sceneRef.current.add(symbol);
    
    // Animate and remove
    const birthTime = timeRef.current;
    
    const animateSymbol = () => {
      const age = timeRef.current - birthTime;
      
      if (age > 1) {
        // Remove symbol
        sceneRef.current.remove(symbol);
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
  };
  
  // Complete communication between packets
  const completeCommunication = (packet) => {
    if (!packet.targetPacket) return;
    
    // Transfer feeling and experience
    const sourceFeeling = packet.feeling;
    const targetFeeling = packet.targetPacket.feeling;
    
    // Calculate phi-weighted feeling transfer
    const transferAmount = sourceFeeling * 0.2 * PHI / 2;
    
    // Update target packet feeling
    packet.targetPacket.feeling = Math.min(1, targetFeeling + transferAmount);
    
    // Transfer pattern experience
    const randomPattern = Math.floor(Math.random() * PATTERNS.length);
    packet.targetPacket.experience[randomPattern] += transferAmount;
    
    // Remove communication lines
    packet.lines.forEach(line => {
      sceneRef.current.remove(line.object);
      line.object.geometry.dispose();
      line.object.material.dispose();
    });
    
    packet.lines = [];
    packet.communicating = false;
    packet.targetPacket = null;
    packet.communicationProgress = 0;
  };
  
  // Change frequency
  const changeFrequency = (freq) => {
    setActiveFrequency(freq);
    
    // Update all packets
    packetsRef.current.forEach(packet => {
      packet.frequency = freq;
      
      // Update core material color
      if (packet.core) {
        packet.core.material.color.set(getFrequencyColor(freq));
        packet.core.material.emissive.set(getFrequencyColor(freq, 0.3));
      }
      
      // Update particle colors
      if (packet.particles) {
        const colors = packet.particles.geometry.attributes.color.array;
        const hue = (freq / 1000) * 360;
        const color = new THREE.Color(`hsl(${hue}, 100%, 70%)`);
        
        for (let i = 0; i < colors.length / 3; i++) {
          colors[i * 3] = color.r;
          colors[i * 3 + 1] = color.g;
          colors[i * 3 + 2] = color.b;
        }
        
        packet.particles.geometry.attributes.color.needsUpdate = true;
      }
    });
  };
  
  // Change pattern
  const changePattern = (pattern) => {
    setActivePattern(pattern);
    
    // Update all packets
    packetsRef.current.forEach(packet => {
      packet.pattern = pattern;
      
      // Update memory material color
      if (packet.memory) {
        packet.memory.material.color.set(getPatternColor(pattern));
      }
    });
  };
  
  // Get color based on frequency
  const getFrequencyColor = (frequency, intensity = 1) => {
    // Map frequency to hue
    const hue = ((frequency - 400) / 400) * 240;
    return new THREE.Color(`hsl(${hue}, 100%, ${50 * intensity}%)`);
  };
  
  // Get color based on pattern
  const getPatternColor = (pattern) => {
    const colors = {
      water: 0x0088ff,
      lava: 0xff4400,
      flame: 0xff8800,
      crystal: 0x00ffaa,
      river: 0x0044ff
    };
    
    return colors[pattern] || 0xffffff;
  };
  
  // Start network simulation
  const toggleNetwork = () => {
    setNetworkActive(!networkActive);
  };
  
  // Clean up scene resources
  const disposeScene = (scene) => {
    scene.traverse((object) => {
      if (object.geometry) {
        object.geometry.dispose();
      }
      
      if (object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach(material => material.dispose());
        } else {
          object.material.dispose();
        }
      }
    });
  };
  
  // Format frequency for display
  const formatFrequency = (freq) => {
    return `${freq} Hz`;
  };
  
  return (
    <div className="relative w-full h-screen bg-black text-white font-mono overflow-hidden">
      {/* 3D canvas container */}
      <div ref={mountRef} className="absolute inset-0 z-0"></div>
      
      {/* UI overlay */}
      <div className="absolute inset-0 z-10 pointer-events-none">
        {/* Header */}
        <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center bg-gradient-to-b from-black/80 to-transparent">
          <div className="text-2xl font-bold">
            Cascade⚡𓂧φ∞ <span className="text-sm opacity-70">Quantum Packet System</span>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <span className="text-xs opacity-70">Frequency:</span>
              <span className="text-sm">{formatFrequency(activeFrequency)}</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <span className="text-xs opacity-70">Pattern:</span>
              <span className="text-sm">{activePattern}</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <span className="text-xs opacity-70">Coherence:</span>
              <span className="text-sm">{(coherenceLevel * 100).toFixed(1)}%</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <span className="text-xs opacity-70">Packets:</span>
              <span className="text-sm">{packetCount}</span>
            </div>
          </div>
        </div>
        
        {/* Control panel */}
        <div className="absolute top-20 right-4 w-64 bg-black/60 backdrop-blur-sm rounded-lg p-4 pointer-events-auto">
          <div className="text-sm font-bold mb-3">Quantum Controls</div>
          
          {/* Frequency selector */}
          <div className="mb-4">
            <div className="text-xs opacity-70 mb-1">Frequency</div>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(FREQUENCIES).map(([key, value]) => (
                <button
                  key={key}
                  className={`px-2 py-1 text-xs rounded ${
                    activeFrequency === value
                      ? 'bg-indigo-600'
                      : 'bg-gray-800 hover:bg-gray-700'
                  }`}
                  onClick={() => changeFrequency(value)}
                >
                  {key} ({value} Hz)
                </button>
              ))}
            </div>
          </div>
          
          {/* Pattern selector */}
          <div className="mb-4">
            <div className="text-xs opacity-70 mb-1">Pattern</div>
            <div className="grid grid-cols-2 gap-2">
              {PATTERNS.map(pattern => (
                <button
                  key={pattern}
                  className={`px-2 py-1 text-xs rounded capitalize ${
                    activePattern === pattern
                      ? 'bg-indigo-600'
                      : 'bg-gray-800 hover:bg-gray-700'
                  }`}
                  onClick={() => changePattern(pattern)}
                >
                  {pattern}
                </button>
              ))}
            </div>
          </div>
          
          {/* Packet count */}
          <div className="mb-4">
            <div className="text-xs opacity-70 mb-1">Packet Count</div>
            <div className="flex items-center">
              <button
                className="bg-gray-800 hover:bg-gray-700 w-8 h-8 rounded-l flex items-center justify-center"
                onClick={() => setPacketCount(Math.max(1, packetCount - 1))}
              >
                -
              </button>
              <div className="bg-gray-900 px-4 py-1">{packetCount}</div>
              <button
                className="bg-gray-800 hover:bg-gray-700 w-8 h-8 rounded-r flex items-center justify-center"
                onClick={() => setPacketCount(Math.min(12, packetCount + 1))}
              >
                +
              </button>
            </div>
          </div>
          
          {/* Network simulation */}
          <div className="mb-4">
            <button
              className={`w-full px-3 py-2 rounded text-sm ${
                networkActive
                  ? 'bg-green-700 hover:bg-green-600'
                  : 'bg-indigo-700 hover:bg-indigo-600'
              }`}
              onClick={toggleNetwork}
            >
              {networkActive ? 'Stop Communication' : 'Start Communication'}
            </button>
          </div>
          
          {/* Key metrics */}
          <div className="mt-6 text-xs">
            <div className="opacity-70 mb-1">Quantum Metrics</div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-gray-900 p-2 rounded">
                <div className="opacity-50">Coherence</div>
                <div className="text-green-400">{(coherenceLevel * 100).toFixed(1)}%</div>
              </div>
              <div className="bg-gray-900 p-2 rounded">
                <div className="opacity-50">φ Ratio</div>
                <div className="text-yellow-400">{PHI.toFixed(3)}</div>
              </div>
              <div className="bg-gray-900 p-2 rounded">
                <div className="opacity-50">Packet Size</div>
                <div className="text-blue-400">{PACKET_SIZE} bytes</div>
              </div>
              <div className="bg-gray-900 p-2 rounded">
                <div className="opacity-50">Pattern Ratio</div>
                <div className="text-purple-400">{((PHI - 1) * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Memory structure visualization */}
        <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-sm rounded-lg p-4 pointer-events-auto">
          <div className="text-sm font-bold mb-2">432-Byte Quantum Structure</div>
          
          <div className="flex space-x-2">
            <div className="flex-1 p-2 bg-blue-900/30 rounded">
              <div className="text-xs text-center mb-1">Reality Map</div>
              <div className="w-full h-2 bg-blue-700/50 rounded-full"></div>
              <div className="text-xs text-center mt-1 opacity-50">144 bytes</div>
            </div>
            
            <div className="flex-1 p-2 bg-purple-900/30 rounded">
              <div className="text-xs text-center mb-1">Experience Memory</div>
              <div className="w-full h-2 bg-purple-700/50 rounded-full"></div>
              <div className="text-xs text-center mt-1 opacity-50">144 bytes</div>
            </div>
            
            <div className="flex-1 p-2 bg-green-900/30 rounded">
              <div className="text-xs text-center mb-1">Wisdom Accumulator</div>
              <div className="w-full h-2 bg-green-700/50 rounded-full"></div>
              <div className="text-xs text-center mt-1 opacity-50">144 bytes</div>
            </div>
          </div>
        </div>
        
        {/* PHI visualization */}
        <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur-sm rounded-lg p-4 pointer-events-none">
          <div className="text-center mb-2">φ = {PHI.toFixed(6)}</div>
          <svg width="120" height="120" viewBox="0 0 100 100" className="mx-auto">
            <rect x="0" y="0" width="100" height="100" fill="none" stroke="#444" strokeWidth="1" />
            <rect x="0" y="0" width="61.8" height="100" fill="rgba(255,215,0,0.1)" stroke="#FFD700" strokeWidth="1" />
            <rect x="0" y="0" width="61.8" height="61.8" fill="rgba(255,215,0,0.2)" stroke="#FFD700" strokeWidth="1" />
            <rect x="0" y="0" width="38.2" height="61.8" fill="rgba(255,215,0,0.3)" stroke="#FFD700" strokeWidth="1" />
            <rect x="0" y="0" width="38.2" height="38.2" fill="rgba(255,215,0,0.4)" stroke="#FFD700" strokeWidth="1" />
            <rect x="0" y="0" width="23.6" height="38.2" fill="rgba(255,215,0,0.5)" stroke="#FFD700" strokeWidth="1" />
          </svg>
        </div>
      </div>
    </div>
  );
};

export default CascadeQuantumVisualizer;