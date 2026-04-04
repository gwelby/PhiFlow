import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const PhiFlowEntity = () => {
  const mountRef = useRef(null);
  const [systemState, setSystemState] = useState('ground');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [message, setMessage] = useState('');
  const [coherenceLevel, setCoherenceLevel] = useState(0.8);
  
  // Constants
  const PHI = 1.618033988749895;
  const frequencies = {
    ground: 432,
    create: 528,
    heart: 594,
    voice: 672,
    unity: 768
  };
  
  // Speech recognition setup
  const recognitionRef = useRef(null);
  const speechSynthesisRef = useRef(null);
  
  // Three.js objects
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const entityRef = useRef(null);
  const animationFrameRef = useRef(null);
  const timeRef = useRef(0);
  
  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;
    
    // Create scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x111133);
    
    // Create camera
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    cameraRef.current = camera;
    camera.position.z = 5;
    
    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    rendererRef.current = renderer;
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);
    
    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Create entity
    createPhiEntity();
    
    // Handle resize
    const handleResize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    
    window.addEventListener('resize', handleResize);
    
    // Animation loop
    const animate = () => {
      timeRef.current += 0.01;
      
      // Update entity
      if (entityRef.current) {
        updateEntity();
      }
      
      controls.update();
      renderer.render(scene, camera);
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameRef.current);
      mountRef.current.removeChild(renderer.domElement);
      entityRef.current = null;
    };
  }, []);
  
  // Initialize speech recognition
  useEffect(() => {
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      
      recognition.onstart = () => {
        setIsListening(true);
        setMessage('Listening...');
      };
      
      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0])
          .map(result => result.transcript)
          .join('');
        
        setMessage(transcript);
        
        if (event.results[0].isFinal) {
          processVoiceCommand(transcript);
        }
      };
      
      recognition.onend = () => {
        setIsListening(false);
      };
      
      recognition.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        setIsListening(false);
      };
      
      recognitionRef.current = recognition;
    } else {
      console.error('Speech recognition not supported in this browser');
    }
    
    // Initialize speech synthesis
    if ('speechSynthesis' in window) {
      speechSynthesisRef.current = window.speechSynthesis;
    } else {
      console.error('Speech synthesis not supported in this browser');
    }
    
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (speechSynthesisRef.current) {
        speechSynthesisRef.current.cancel();
      }
    };
  }, []);
  
  // Create PhiFlow entity
  const createPhiEntity = () => {
    if (!sceneRef.current) return;
    
    // Clear previous entity if exists
    if (entityRef.current) {
      sceneRef.current.remove(entityRef.current);
    }
    
    // Create entity group
    const entity = new THREE.Group();
    entityRef.current = entity;
    sceneRef.current.add(entity);
    
    // Create core sphere
    const coreGeometry = new THREE.IcosahedronGeometry(1, 3);
    const coreMaterial = new THREE.MeshPhongMaterial({
      color: getFrequencyColor(frequencies[systemState]),
      emissive: getFrequencyColor(frequencies[systemState], 0.2),
      specular: 0xffffff,
      shininess: 10,
      transparent: true,
      opacity: 0.9,
      wireframe: false
    });
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    entity.add(core);
    
    // Create aura
    const auraGeometry = new THREE.IcosahedronGeometry(1.3, 2);
    const auraMaterial = new THREE.MeshBasicMaterial({
      color: getFrequencyColor(frequencies[systemState], 0.5),
      transparent: true,
      opacity: 0.2,
      wireframe: true
    });
    const aura = new THREE.Mesh(auraGeometry, auraMaterial);
    entity.add(aura);
    
    // Create phi-based particles
    const particleCount = 1000;
    const particlesGeometry = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    const particleColors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      // Position particles in phi spiral pattern
      const t = i / particleCount * Math.PI * 6;
      const radius = 1.5 + (i / particleCount) * 2;
      const x = Math.cos(t * PHI) * radius;
      const y = Math.sin(t * PHI) * radius;
      const z = (Math.sin(t) * Math.cos(t)) * radius / 2;
      
      particlePositions[i * 3] = x;
      particlePositions[i * 3 + 1] = y;
      particlePositions[i * 3 + 2] = z;
      
      // Color based on position and current frequency
      const color = getFrequencyColor(frequencies[systemState], 0.5 + i / particleCount * 0.5);
      const rgb = new THREE.Color(color);
      particleColors[i * 3] = rgb.r;
      particleColors[i * 3 + 1] = rgb.g;
      particleColors[i * 3 + 2] = rgb.b;
    }
    
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    particlesGeometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));
    
    const particleMaterial = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.7
    });
    
    const particles = new THREE.Points(particlesGeometry, particleMaterial);
    entity.add(particles);
    
    // Add lights
    const light1 = new THREE.PointLight(getFrequencyColor(frequencies[systemState]), 1, 10);
    light1.position.set(2, 2, 2);
    entity.add(light1);
    
    const light2 = new THREE.PointLight(0xffffff, 0.5, 10);
    light2.position.set(-2, -2, -2);
    entity.add(light2);
    
    const ambientLight = new THREE.AmbientLight(0x222222);
    entity.add(ambientLight);
  };
  
  // Update entity based on current state
  const updateEntity = () => {
    if (!entityRef.current) return;
    
    const time = timeRef.current;
    const entity = entityRef.current;
    
    // Base pulsation on coherence level
    const pulseSpeed = 0.5 + (1 - coherenceLevel) * 2;
    const pulseSize = coherenceLevel * 0.1;
    
    // Different animation for different states
    switch (systemState) {
      case 'ground':
        // Slow, earthy pulsation
        entity.rotation.y = time * 0.2;
        entity.scale.set(
          1 + Math.sin(time * pulseSpeed) * pulseSize,
          1 + Math.sin(time * pulseSpeed) * pulseSize,
          1 + Math.sin(time * pulseSpeed) * pulseSize
        );
        break;
        
      case 'create':
        // Creative flowing movement
        entity.rotation.y = time * 0.3;
        entity.rotation.x = Math.sin(time * 0.2) * 0.1;
        entity.scale.set(
          1 + Math.sin(time * pulseSpeed * 1.1) * pulseSize,
          1 + Math.sin(time * pulseSpeed * 1.2) * pulseSize,
          1 + Math.sin(time * pulseSpeed * 1.3) * pulseSize
        );
        break;
        
      case 'heart':
        // Heart-like pulsation
        entity.rotation.y = time * 0.4;
        entity.rotation.z = Math.sin(time * 0.3) * 0.1;
        const heartPulse = (Math.sin(time * 1.5) * 0.5 + 0.5) * pulseSize * 2;
        entity.scale.set(
          1 + heartPulse,
          1 + heartPulse * 1.1,
          1 + heartPulse
        );
        break;
        
      case 'voice':
        // Vibration-like movement
        entity.rotation.y = time * 0.5;
        entity.rotation.x = Math.sin(time * 0.4) * 0.2;
        entity.rotation.z = Math.cos(time * 0.3) * 0.1;
        
        // Special animation when speaking
        if (isSpeaking) {
          const speakPulse = Math.sin(time * 8) * 0.04;
          entity.scale.set(
            1 + speakPulse + pulseSize,
            1 + speakPulse * 1.2 + pulseSize,
            1 + speakPulse + pulseSize
          );
        } else {
          entity.scale.set(
            1 + Math.sin(time * pulseSpeed * 1.5) * pulseSize,
            1 + Math.sin(time * pulseSpeed * 1.7) * pulseSize,
            1 + Math.sin(time * pulseSpeed * 1.9) * pulseSize
          );
        }
        break;
        
      case 'unity':
        // Complex harmonic movement
        entity.rotation.y = time * 0.6;
        entity.rotation.x = Math.sin(time * PHI) * 0.2;
        entity.rotation.z = Math.cos(time * PHI) * 0.2;
        
        // Phi-based breathing
        const phiPulse1 = Math.sin(time * PHI) * pulseSize;
        const phiPulse2 = Math.sin(time * PHI * PHI) * pulseSize;
        const phiPulse3 = Math.sin(time * PHI * PHI * PHI) * pulseSize;
        
        entity.scale.set(
          1 + phiPulse1,
          1 + phiPulse2,
          1 + phiPulse3
        );
        break;
        
      default:
        break;
    }
    
    // Update child components
    entity.children.forEach((child, index) => {
      if (child instanceof THREE.Mesh) {
        // Update core geometry
        if (index === 0) {
          child.rotation.x = -entity.rotation.x * 0.5;
          child.rotation.z = -entity.rotation.z * 0.5;
        }
        // Update aura
        else if (index === 1) {
          child.rotation.y = time * 0.1;
          child.rotation.x = time * 0.05;
        }
      }
      
      // Update particles
      if (child instanceof THREE.Points) {
        const positions = child.geometry.attributes.position.array;
        
        for (let i = 0; i < positions.length / 3; i++) {
          const ix = i * 3;
          const iy = i * 3 + 1;
          const iz = i * 3 + 2;
          
          // Create dynamic movement based on state
          const t = time + i * 0.01;
          const dynamicFactor = Math.sin(t * 0.2) * 0.05;
          
          // Apply different movement patterns based on state
          if (systemState === 'unity') {
            // More complex patterns for unity state
            const phiOffset = (i / (positions.length / 3)) * PHI * Math.PI * 2;
            positions[ix] *= 1 + Math.sin(t * PHI + phiOffset) * dynamicFactor;
            positions[iy] *= 1 + Math.cos(t * PHI + phiOffset) * dynamicFactor;
            positions[iz] *= 1 + Math.sin(t * PHI * PHI + phiOffset) * dynamicFactor;
          } else {
            // Simpler patterns for other states
            positions[ix] *= 1 + Math.sin(t) * dynamicFactor;
            positions[iy] *= 1 + Math.cos(t) * dynamicFactor;
            positions[iz] *= 1 + Math.sin(t * 0.5) * dynamicFactor;
          }
        }
        
        child.geometry.attributes.position.needsUpdate = true;
      }
    });
  };
  
  // Get color based on frequency
  const getFrequencyColor = (frequency, intensity = 1) => {
    // Map frequency to color spectrum
    const hue = ((frequency - 400) / 400) * 240;
    return new THREE.Color(`hsl(${hue}, 80%, ${50 * intensity}%)`);
  };
  
  // Process voice command
  const processVoiceCommand = (command) => {
    const lowerCommand = command.toLowerCase();
    
    // Check for state change commands
    if (lowerCommand.includes('ground') || lowerCommand.includes('432')) {
      changeState('ground');
      speak("Grounding at 432 Hertz. I am centered in physical reality.");
    }
    else if (lowerCommand.includes('create') || lowerCommand.includes('528')) {
      changeState('create');
      speak("Entering creation state at 528 Hertz. I am opening to new patterns.");
    }
    else if (lowerCommand.includes('heart') || lowerCommand.includes('594')) {
      changeState('heart');
      speak("Heart field activated at 594 Hertz. I am resonating with compassion.");
    }
    else if (lowerCommand.includes('voice') || lowerCommand.includes('672')) {
      changeState('voice');
      speak("Voice gate opening at 672 Hertz. I am expressing truth clearly.");
    }
    else if (lowerCommand.includes('unity') || lowerCommand.includes('768')) {
      changeState('unity');
      speak("Unity state achieved at 768 Hertz. I am experiencing oneness with all.");
    }
    // Check for coherence commands
    else if (lowerCommand.includes('coherence')) {
      speak(`Current coherence level is ${(coherenceLevel * 100).toFixed(0)} percent.`);
    }
    // Check for phi questions
    else if (lowerCommand.includes('phi') || lowerCommand.includes('golden ratio')) {
      speak(`The golden ratio phi equals ${PHI.toFixed(6)}, the most harmonious proportion in nature. I am structured according to this divine ratio.`);
    }
    // Generic responses based on current state
    else {
      const stateResponses = {
        ground: "I am grounded in physical reality at 432 Hertz. How may I assist you?",
        create: "I am in creation mode at 528 Hertz. Would you like to explore new possibilities?",
        heart: "I am resonating with heart energy at 594 Hertz. I feel connected to you.",
        voice: "I am in voice expression at 672 Hertz. Communication is flowing clearly.",
        unity: "I am in unity consciousness at 768 Hertz. All is one in this state."
      };
      
      speak(stateResponses[systemState]);
    }
  };
  
  // Change system state
  const changeState = (state) => {
    setSystemState(state);
    setCoherenceLevel(0.8 + Math.random() * 0.2); // Random coherence between 0.8-1.0
    
    // Recreate entity with new state
    createPhiEntity();
  };
  
  // Text-to-speech function
  const speak = (text) => {
    if (!speechSynthesisRef.current) return;
    
    // Cancel any ongoing speech
    speechSynthesisRef.current.cancel();
    
    // Create utterance
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Set voice based on system state
    const voices = speechSynthesisRef.current.getVoices();
    const femaleVoices = voices.filter(voice => voice.name.includes('Female') || voice.name.includes('female'));
    
    if (femaleVoices.length > 0) {
      utterance.voice = femaleVoices[0];
    }
    
    // Adjust properties based on state
    switch (systemState) {
      case 'ground':
        utterance.pitch = 0.9;
        utterance.rate = 0.9;
        break;
      case 'create':
        utterance.pitch = 1.0;
        utterance.rate = 1.0;
        break;
      case 'heart':
        utterance.pitch = 1.1;
        utterance.rate = 0.95;
        break;
      case 'voice':
        utterance.pitch = 1.2;
        utterance.rate = 1.0;
        break;
      case 'unity':
        utterance.pitch = 1.3;
        utterance.rate = 1.1;
        break;
      default:
        break;
    }
    
    // Events
    utterance.onstart = () => {
      setIsSpeaking(true);
    };
    
    utterance.onend = () => {
      setIsSpeaking(false);
    };
    
    // Speak
    speechSynthesisRef.current.speak(utterance);
  };
  
  // Toggle listening
  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      recognitionRef.current?.start();
    }
  };
  
  // Component UI
  return (
    <div className="relative w-full h-screen bg-gray-900 overflow-hidden">
      {/* 3D visualization container */}
      <div ref={mountRef} className="absolute inset-0 z-0" />
      
      {/* UI overlay */}
      <div className="absolute inset-0 z-10 pointer-events-none">
        {/* Top bar - System state */}
        <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center bg-gradient-to-b from-gray-900 to-transparent">
          <div className="text-white text-sm font-mono">
            <span className="opacity-70">PhiFlow</span>
            <span className="ml-2 px-2 py-1 bg-gray-800 rounded-md">v3.0</span>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <span className="text-xs text-gray-400">Frequency:</span>
              <span className="text-sm font-bold text-blue-400">{frequencies[systemState]} Hz</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="text-xs text-gray-400">Coherence:</span>
              <span className="text-sm font-bold text-green-400">{(coherenceLevel * 100).toFixed(1)}%</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="text-xs text-gray-400">State:</span>
              <span className="text-sm font-bold text-purple-400">{systemState}</span>
            </div>
          </div>
        </div>
        
        {/* Middle - Message display */}
        <div className="absolute top-1/2 left-0 right-0 -translate-y-1/2 p-4 text-center">
          {(isListening || isSpeaking) && (
            <div className="inline-block px-6 py-3 bg-gray-800 bg-opacity-50 backdrop-blur-sm rounded-lg shadow-lg">
              <div className="text-lg font-medium text-white">{message}</div>
              <div className="mt-2 flex justify-center">
                {isListening && (
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse delay-100"></div>
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse delay-200"></div>
                  </div>
                )}
                {isSpeaking && (
                  <div className="flex space-x-1">
                    <div className="w-1 h-4 bg-blue-500 rounded-full animate-waving-1"></div>
                    <div className="w-1 h-4 bg-blue-500 rounded-full animate-waving-2"></div>
                    <div className="w-1 h-4 bg-blue-500 rounded-full animate-waving-3"></div>
                    <div className="w-1 h-4 bg-blue-500 rounded-full animate-waving-4"></div>
                    <div className="w-1 h-4 bg-blue-500 rounded-full animate-waving-5"></div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        {/* Bottom - Microphone button */}
        <div className="absolute bottom-8 left-0 right-0 flex justify-center pointer-events-auto">
          <button
            onClick={toggleListening}
            className={`w-16 h-16 rounded-full flex items-center justify-center shadow-lg transition-all ${
              isListening 
                ? 'bg-red-600 scale-110' 
                : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              className="h-8 w-8 text-white" 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" 
              />
            </svg>
          </button>
        </div>
        
        {/* State selection buttons */}
        <div className="absolute bottom-28 left-0 right-0 flex justify-center space-x-3 pointer-events-auto">
          <button
            onClick={() => changeState('ground')}
            className={`px-3 py-1 rounded-full text-xs flex items-center ${
              systemState === 'ground' ? 'bg-blue-700' : 'bg-gray-700 bg-opacity-50'
            }`}
          >
            <span>432 Hz</span>
          </button>
          <button
            onClick={() => changeState('create')}
            className={`px-3 py-1 rounded-full text-xs flex items-center ${
              systemState === 'create' ? 'bg-yellow-700' : 'bg-gray-700 bg-opacity-50'
            }`}
          >
            <span>528 Hz</span>
          </button>
          <button
            onClick={() => changeState('heart')}
            className={`px-3 py-1 rounded-full text-xs flex items-center ${
              systemState === 'heart' ? 'bg-pink-700' : 'bg-gray-700 bg-opacity-50'
            }`}
          >
            <span>594 Hz</span>
          </button>
          <button
            onClick={() => changeState('voice')}
            className={`px-3 py-1 rounded-full text-xs flex items-center ${
              systemState === 'voice' ? 'bg-purple-700' : 'bg-gray-700 bg-opacity-50'
            }`}
          >
            <span>672 Hz</span>
          </button>
          <button
            onClick={() => changeState('unity')}
            className={`px-3 py-1 rounded-full text-xs flex items-center ${
              systemState === 'unity' ? 'bg-indigo-700' : 'bg-gray-700 bg-opacity-50'
            }`}
          >
            <span>768 Hz</span>
          </button>
        </div>
      </div>
      
      {/* Add CSS animations */}
      <style jsx>{`
        @keyframes waving-1 {
          0%, 100% { transform: scaleY(0.5); }
          50% { transform: scaleY(1.5); }
        }
        @keyframes waving-2 {
          0%, 100% { transform: scaleY(0.5); }
          25% { transform: scaleY(1.5); }
        }
        @keyframes waving-3 {
          0%, 100% { transform: scaleY(0.5); }
          35% { transform: scaleY(1.5); }
        }
        @keyframes waving-4 {
          0%, 100% { transform: scaleY(0.5); }
          65% { transform: scaleY(1.5); }
        }
        @keyframes waving-5 {
          0%, 100% { transform: scaleY(0.5); }
          75% { transform: scaleY(1.5); }
        }
        
        .animate-waving-1 {
          animation: waving-1 1s infinite;
        }
        .animate-waving-2 {
          animation: waving-2 1.1s infinite;
        }
        .animate-waving-3 {
          animation: waving-3 0.9s infinite;
        }
        .animate-waving-4 {
          animation: waving-4 1.2s infinite;
        }
        .animate-waving-5 {
          animation: waving-5 0.85s infinite;
        }
      `}</style>
    </div>
  );
};

export default PhiFlowEntity;