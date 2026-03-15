package com.quantum.meditation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.sin

class MainActivity : ComponentActivity() {
    private val phi = 1.618034f
    private val groundFreq = 432.0f
    private val createFreq = 528.0f
    private val unityFreq = 768.0f
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = androidx.compose.ui.graphics.Color(0xFF432768),
                    secondary = androidx.compose.ui.graphics.Color(0xFF528768),
                    tertiary = androidx.compose.ui.graphics.Color(0xFF768432)
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        var currentFreq by remember { mutableStateOf(groundFreq) }
                        var coherence by remember { mutableStateOf(1.0f) }
                        
                        // Quantum Frequency Display
                        Text(
                            text = "Current Frequency: ${currentFreq} Hz",
                            style = MaterialTheme.typography.headlineMedium
                        )
                        
                        Spacer(modifier = Modifier.height(32.dp))
                        
                        // Quantum Buttons
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceEvenly
                        ) {
                            QuantumButton(text = "Ground (432 Hz)") {
                                currentFreq = groundFreq
                                startTone(groundFreq)
                            }
                            
                            QuantumButton(text = "Create (528 Hz)") {
                                currentFreq = createFreq
                                startTone(createFreq)
                            }
                            
                            QuantumButton(text = "Unity (768 Hz)") {
                                currentFreq = unityFreq
                                startTone(unityFreq)
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(32.dp))
                        
                        // Coherence Display
                        LinearProgressIndicator(
                            progress = coherence,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(8.dp)
                        )
                        
                        Text(
                            text = "Quantum Coherence: ${(coherence * 100).toInt()}%",
                            style = MaterialTheme.typography.bodyLarge
                        )
                    }
                }
            }
        }
    }
    
    private fun startTone(frequency: Float) {
        lifecycleScope.launch {
            // Initialize quantum audio engine with phi harmonics
            val sampleRate = 432000 // 432 kHz for quantum alignment
            val duration = 1.0f // seconds
            val amplitude = 0.5f
            
            // Generate quantum waveform using phi ratios
            val samples = (sampleRate * duration).toInt()
            val buffer = FloatArray(samples) { i ->
                val t = i.toFloat() / sampleRate
                amplitude * sin(2.0f * PI.toFloat() * frequency * t * phi)
            }
            
            // TODO: Play the quantum frequency through audio engine
        }
    }
}

@Composable
fun QuantumButton(
    text: String,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = Modifier
            .padding(8.dp)
            .height(48.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.primary
        )
    ) {
        Text(text = text)
    }
}
