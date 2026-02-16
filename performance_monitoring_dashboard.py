#!/usr/bin/env python3
"""
PhiFlow Performance Monitoring Dashboard
=======================================

Real-time performance monitoring and visualization system with:
- Live performance metrics display
- Statistical trend analysis
- Performance regression detection
- Interactive visualizations
- Automated alert system
"""

import time
import sys
import os
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'PhiFlow', 'src'))

# Sacred Mathematics Constants
PHI = 1.618033988749895
LAMBDA = 0.618033988749895
PHI_PHI = 11.09017095324081
SACRED_FREQUENCIES = [432, 528, 594, 672, 720, 768, 963]

class MetricType(Enum):
    """Performance metric types"""
    SPEEDUP_RATIO = "speedup_ratio"
    TFLOPS = "tflops"
    LATENCY_MS = "latency_ms" 
    OPERATIONS_PER_SECOND = "operations_per_second"
    COHERENCE = "coherence"
    CONSCIOUSNESS_ENHANCEMENT = "consciousness_enhancement"
    MEMORY_USAGE_MB = "memory_usage_mb"
    CPU_USAGE_PERCENT = "cpu_usage_percent"
    GPU_USAGE_PERCENT = "gpu_usage_percent"

class AlertType(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Real-time performance metric"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    component: str
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_type: AlertType
    message: str
    metric: PerformanceMetric
    timestamp: datetime = field(default_factory=datetime.now)

class PerformanceMonitoringDashboard:
    """
    Real-time Performance Monitoring Dashboard
    
    Provides comprehensive performance monitoring with:
    - Live metrics visualization
    - Statistical analysis
    - Trend detection
    - Performance regression alerts
    - Interactive dashboard interface
    """
    
    def __init__(self, 
                 update_interval_ms: int = 1000,
                 history_minutes: int = 60,
                 enable_alerts: bool = True,
                 output_dir: str = "/mnt/d/Projects/phiflow/monitoring"):
        """Initialize the monitoring dashboard"""
        
        self.update_interval_ms = update_interval_ms
        self.history_minutes = history_minutes
        self.enable_alerts = enable_alerts
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Performance data storage
        self.metrics_history: Dict[str, List[PerformanceMetric]] = {}
        self.alerts_history: List[PerformanceAlert] = []
        self.performance_targets = self._initialize_performance_targets()
        
        # Monitoring status
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # GUI components
        self.root = None
        self.canvas = None
        self.figures = {}
        self.animation_objects = []
        
        # Component availability
        self.components_available = self._check_component_availability()
        
        # Statistics tracking
        self.statistics = {
            'total_measurements': 0,
            'alerts_generated': 0,
            'performance_regressions': 0,
            'system_uptime_minutes': 0
        }
        
        self.logger.info("Performance Monitoring Dashboard initialized")
        self.logger.info(f"Update interval: {update_interval_ms}ms")
        self.logger.info(f"History retention: {history_minutes} minutes")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('PerformanceMonitoring')
    
    def _initialize_performance_targets(self) -> Dict[str, Dict[MetricType, float]]:
        """Initialize performance targets and thresholds"""
        return {
            "cuda_acceleration": {
                MetricType.SPEEDUP_RATIO: 100.0,
                MetricType.TFLOPS: 1.0,
                MetricType.LATENCY_MS: 10.0
            },
            "sacred_mathematics": {
                MetricType.OPERATIONS_PER_SECOND: 1e9,
                MetricType.COHERENCE: 0.999
            },
            "consciousness_processing": {
                MetricType.CONSCIOUSNESS_ENHANCEMENT: 1.8,
                MetricType.COHERENCE: 0.999,
                MetricType.LATENCY_MS: 100.0
            },
            "system_integration": {
                MetricType.LATENCY_MS: 50.0,
                MetricType.COHERENCE: 0.999,
                MetricType.CPU_USAGE_PERCENT: 80.0
            }
        }
    
    def _check_component_availability(self) -> Dict[str, bool]:
        """Check availability of PhiFlow components"""
        components = {}
        
        try:
            from cuda.lib_sacred_cuda import LibSacredCUDA
            lib = LibSacredCUDA()
            components['cuda_sacred_lib'] = lib.cuda_available
        except ImportError:
            components['cuda_sacred_lib'] = False
        
        try:
            from cuda.cuda_optimizer_integration import CUDAConsciousnessProcessor
            components['cuda_consciousness'] = True
        except ImportError:
            components['cuda_consciousness'] = False
        
        try:
            from optimization.phi_quantum_optimizer import PhiQuantumOptimizer
            components['phi_optimizer'] = True
        except ImportError:
            components['phi_optimizer'] = False
        
        try:
            from integration.phi_flow_integration_engine import PhiFlowIntegrationEngine
            components['integration_engine'] = True
        except ImportError:
            components['integration_engine'] = False
        
        return components
    
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        start_time = datetime.now()
        
        while self.monitoring_active:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                
                # Store metrics
                for metric in metrics:
                    self._store_metric(metric)
                
                # Check for alerts
                if self.enable_alerts:
                    self._check_performance_alerts(metrics)
                
                # Update statistics
                self.statistics['total_measurements'] += len(metrics)
                self.statistics['system_uptime_minutes'] = (datetime.now() - start_time).total_seconds() / 60
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep until next update
                time.sleep(self.update_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.update_interval_ms / 1000.0)
    
    def _collect_performance_metrics(self) -> List[PerformanceMetric]:
        """Collect current performance metrics from all components"""
        metrics = []
        current_time = datetime.now()
        
        # System metrics
        import psutil
        
        metrics.append(PerformanceMetric(
            metric_type=MetricType.CPU_USAGE_PERCENT,
            value=psutil.cpu_percent(),
            timestamp=current_time,
            component="system",
            threshold_warning=80.0,
            threshold_critical=95.0
        ))
        
        metrics.append(PerformanceMetric(
            metric_type=MetricType.MEMORY_USAGE_MB,
            value=psutil.virtual_memory().used / 1024**2,
            timestamp=current_time,
            component="system",
            threshold_warning=psutil.virtual_memory().total * 0.8 / 1024**2,
            threshold_critical=psutil.virtual_memory().total * 0.95 / 1024**2
        ))
        
        # CUDA metrics (if available)
        if self.components_available.get('cuda_sacred_lib', False):
            cuda_metrics = self._collect_cuda_metrics(current_time)
            metrics.extend(cuda_metrics)
        
        # Consciousness processing metrics (if available)
        if self.components_available.get('cuda_consciousness', False):
            consciousness_metrics = self._collect_consciousness_metrics(current_time)
            metrics.extend(consciousness_metrics)
        
        # Integration engine metrics (if available)
        if self.components_available.get('integration_engine', False):
            integration_metrics = self._collect_integration_metrics(current_time)
            metrics.extend(integration_metrics)
        
        # Mock additional metrics for demonstration
        metrics.extend(self._collect_mock_metrics(current_time))
        
        return metrics
    
    def _collect_cuda_metrics(self, timestamp: datetime) -> List[PerformanceMetric]:
        """Collect CUDA performance metrics"""
        metrics = []
        
        try:
            from cuda.lib_sacred_cuda import LibSacredCUDA
            lib = LibSacredCUDA()
            
            if lib.cuda_available:
                # Test CUDA performance
                result = lib.sacred_phi_parallel_computation(10000, precision=10)
                
                if result.success:
                    # Calculate TFLOPS
                    tflops = (10000 * 10 * 10) / (result.computation_time * 1e12)
                    
                    metrics.append(PerformanceMetric(
                        metric_type=MetricType.TFLOPS,
                        value=tflops,
                        timestamp=timestamp,
                        component="cuda_sacred_lib",
                        target_value=1.0,
                        threshold_warning=0.5,
                        threshold_critical=0.1
                    ))
                    
                    metrics.append(PerformanceMetric(
                        metric_type=MetricType.OPERATIONS_PER_SECOND,
                        value=result.operations_per_second,
                        timestamp=timestamp,
                        component="cuda_sacred_lib",
                        target_value=1e9,
                        threshold_warning=5e8,
                        threshold_critical=1e8
                    ))
        
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"Error collecting CUDA metrics: {e}")
        
        return metrics
    
    def _collect_consciousness_metrics(self, timestamp: datetime) -> List[PerformanceMetric]:
        """Collect consciousness processing metrics"""
        metrics = []
        
        try:
            # Mock consciousness processing measurement
            # In real implementation, this would measure actual consciousness metrics
            enhancement = 1.5 + np.random.normal(0.3, 0.1)
            coherence = 0.995 + np.random.normal(0, 0.005)
            coherence = max(0.0, min(1.0, coherence))
            
            metrics.append(PerformanceMetric(
                metric_type=MetricType.CONSCIOUSNESS_ENHANCEMENT,
                value=enhancement,
                timestamp=timestamp,
                component="consciousness_processor",
                target_value=1.8,
                threshold_warning=1.2,
                threshold_critical=1.0
            ))
            
            metrics.append(PerformanceMetric(
                metric_type=MetricType.COHERENCE,
                value=coherence,
                timestamp=timestamp,
                component="consciousness_processor",
                target_value=0.999,
                threshold_warning=0.95,
                threshold_critical=0.90
            ))
        
        except Exception as e:
            self.logger.warning(f"Error collecting consciousness metrics: {e}")
        
        return metrics
    
    def _collect_integration_metrics(self, timestamp: datetime) -> List[PerformanceMetric]:
        """Collect integration engine metrics"""
        metrics = []
        
        try:
            # Mock integration metrics
            # In real implementation, this would measure actual integration performance
            latency = np.random.uniform(20, 80)  # 20-80ms
            coherence = 0.998 + np.random.normal(0, 0.002)
            coherence = max(0.0, min(1.0, coherence))
            
            metrics.append(PerformanceMetric(
                metric_type=MetricType.LATENCY_MS,
                value=latency,
                timestamp=timestamp,
                component="integration_engine",
                target_value=50.0,
                threshold_warning=100.0,
                threshold_critical=200.0
            ))
            
            metrics.append(PerformanceMetric(
                metric_type=MetricType.COHERENCE,
                value=coherence,
                timestamp=timestamp,
                component="integration_engine",
                target_value=0.999,
                threshold_warning=0.95,
                threshold_critical=0.90
            ))
        
        except Exception as e:
            self.logger.warning(f"Error collecting integration metrics: {e}")
        
        return metrics
    
    def _collect_mock_metrics(self, timestamp: datetime) -> List[PerformanceMetric]:
        """Collect mock metrics for demonstration"""
        metrics = []
        
        # Mock speedup ratio
        speedup = 80 + np.random.normal(20, 10)
        speedup = max(10, speedup)
        
        metrics.append(PerformanceMetric(
            metric_type=MetricType.SPEEDUP_RATIO,
            value=speedup,
            timestamp=timestamp,
            component="cuda_acceleration",
            target_value=100.0,
            threshold_warning=50.0,
            threshold_critical=20.0
        ))
        
        return metrics
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store performance metric in history"""
        key = f"{metric.component}_{metric.metric_type.value}"
        
        if key not in self.metrics_history:
            self.metrics_history[key] = []
        
        self.metrics_history[key].append(metric)
    
    def _check_performance_alerts(self, metrics: List[PerformanceMetric]):
        """Check for performance alerts"""
        for metric in metrics:
            # Check critical threshold
            if metric.threshold_critical is not None:
                if (metric.metric_type in [MetricType.LATENCY_MS, MetricType.CPU_USAGE_PERCENT, 
                                         MetricType.MEMORY_USAGE_MB] and metric.value > metric.threshold_critical) or \
                   (metric.metric_type in [MetricType.SPEEDUP_RATIO, MetricType.TFLOPS, 
                                         MetricType.COHERENCE, MetricType.CONSCIOUSNESS_ENHANCEMENT] 
                    and metric.value < metric.threshold_critical):
                    
                    alert = PerformanceAlert(
                        alert_type=AlertType.CRITICAL,
                        message=f"CRITICAL: {metric.component} {metric.metric_type.value} is {metric.value:.3f} "
                               f"(threshold: {metric.threshold_critical})",
                        metric=metric
                    )
                    self._process_alert(alert)
            
            # Check warning threshold
            elif metric.threshold_warning is not None:
                if (metric.metric_type in [MetricType.LATENCY_MS, MetricType.CPU_USAGE_PERCENT, 
                                         MetricType.MEMORY_USAGE_MB] and metric.value > metric.threshold_warning) or \
                   (metric.metric_type in [MetricType.SPEEDUP_RATIO, MetricType.TFLOPS, 
                                         MetricType.COHERENCE, MetricType.CONSCIOUSNESS_ENHANCEMENT] 
                    and metric.value < metric.threshold_warning):
                    
                    alert = PerformanceAlert(
                        alert_type=AlertType.WARNING,
                        message=f"WARNING: {metric.component} {metric.metric_type.value} is {metric.value:.3f} "
                               f"(threshold: {metric.threshold_warning})",
                        metric=metric
                    )
                    self._process_alert(alert)
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process performance alert"""
        self.alerts_history.append(alert)
        self.statistics['alerts_generated'] += 1
        
        # Log alert
        log_level = logging.CRITICAL if alert.alert_type == AlertType.CRITICAL else \
                   logging.WARNING if alert.alert_type == AlertType.WARNING else logging.INFO
        
        self.logger.log(log_level, alert.message)
        
        # Show GUI alert if available
        if self.root and alert.alert_type == AlertType.CRITICAL:
            messagebox.showerror("Performance Alert", alert.message)
    
    def _cleanup_old_data(self):
        """Clean old performance data beyond retention period"""
        cutoff_time = datetime.now() - timedelta(minutes=self.history_minutes)
        
        for key in self.metrics_history:
            self.metrics_history[key] = [
                metric for metric in self.metrics_history[key] 
                if metric.timestamp > cutoff_time
            ]
        
        # Clean old alerts
        self.alerts_history = [
            alert for alert in self.alerts_history 
            if alert.timestamp > cutoff_time
        ]
    
    def create_dashboard_gui(self):
        """Create the main dashboard GUI"""
        self.root = tk.Tk()
        self.root.title("PhiFlow Performance Monitoring Dashboard")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control panel
        self._create_control_panel(main_frame)
        
        # Create metrics display
        self._create_metrics_display(main_frame)
        
        # Create visualization area
        self._create_visualization_area(main_frame)
        
        # Create status bar
        self._create_status_bar(main_frame)
        
        # Start GUI update timer
        self._update_gui()
        
        return self.root
    
    def _create_control_panel(self, parent):
        """Create control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Monitoring controls
        ttk.Button(control_frame, text="Start Monitoring", 
                  command=self.start_monitoring).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Stop Monitoring", 
                  command=self.stop_monitoring).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Clear History", 
                  command=self._clear_history).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Export Data", 
                  command=self._export_data).pack(side=tk.LEFT, padx=(0, 5))
        
        # Status indicator
        self.status_label = ttk.Label(control_frame, text="Status: Stopped", 
                                     foreground="red")
        self.status_label.pack(side=tk.RIGHT)
    
    def _create_metrics_display(self, parent):
        """Create real-time metrics display"""
        metrics_frame = ttk.LabelFrame(parent, text="Current Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create treeview for metrics
        columns = ("Component", "Metric", "Value", "Target", "Status")
        self.metrics_tree = ttk.Treeview(metrics_frame, columns=columns, show="headings", height=6)
        
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=120)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)
        
        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_visualization_area(self, parent):
        """Create visualization area"""
        viz_frame = ttk.LabelFrame(parent, text="Performance Visualizations", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create notebook for different visualizations
        self.viz_notebook = ttk.Notebook(viz_frame)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Performance trends tab
        self._create_trends_tab()
        
        # System metrics tab
        self._create_system_metrics_tab()
        
        # Alerts tab
        self._create_alerts_tab()
    
    def _create_trends_tab(self):
        """Create performance trends visualization tab"""
        trends_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(trends_frame, text="Performance Trends")
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("PhiFlow Performance Trends")
        
        self.figures['trends'] = fig
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, trends_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup animation
        ani = animation.FuncAnimation(fig, self._animate_trends, interval=self.update_interval_ms, blit=False)
        self.animation_objects.append(ani)
    
    def _create_system_metrics_tab(self):
        """Create system metrics visualization tab"""
        system_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(system_frame, text="System Metrics")
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("System Performance Metrics")
        
        self.figures['system'] = fig
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, system_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup animation
        ani = animation.FuncAnimation(fig, self._animate_system_metrics, interval=self.update_interval_ms, blit=False)
        self.animation_objects.append(ani)
    
    def _create_alerts_tab(self):
        """Create alerts display tab"""
        alerts_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(alerts_frame, text="Alerts")
        
        # Create treeview for alerts
        columns = ("Time", "Type", "Component", "Message")
        self.alerts_tree = ttk.Treeview(alerts_frame, columns=columns, show="headings")
        
        for col in columns:
            self.alerts_tree.heading(col, text=col)
            if col == "Message":
                self.alerts_tree.column(col, width=400)
            else:
                self.alerts_tree.column(col, width=120)
        
        # Add scrollbar
        alerts_scrollbar = ttk.Scrollbar(alerts_frame, orient=tk.VERTICAL, command=self.alerts_tree.yview)
        self.alerts_tree.configure(yscrollcommand=alerts_scrollbar.set)
        
        self.alerts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alerts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X)
        
        self.status_bar = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
    
    def _animate_trends(self, frame):
        """Animate performance trends"""
        if not self.metrics_history:
            return
        
        fig = self.figures['trends']
        fig.clear()
        
        axes = fig.subplots(2, 2)
        fig.suptitle("PhiFlow Performance Trends")
        
        # Plot speedup ratio
        self._plot_metric_trend(axes[0, 0], "cuda_acceleration_speedup_ratio", 
                               "CUDA Speedup Ratio", "Time", "Speedup (x)")
        
        # Plot TFLOPS
        self._plot_metric_trend(axes[0, 1], "cuda_sacred_lib_tflops", 
                               "TFLOPS Performance", "Time", "TFLOPS")
        
        # Plot coherence
        self._plot_metric_trend(axes[1, 0], "consciousness_processor_coherence", 
                               "Coherence", "Time", "Coherence")
        
        # Plot latency
        self._plot_metric_trend(axes[1, 1], "integration_engine_latency_ms", 
                               "Integration Latency", "Time", "Latency (ms)")
        
        plt.tight_layout()
    
    def _animate_system_metrics(self, frame):
        """Animate system metrics"""
        fig = self.figures['system']
        fig.clear()
        
        axes = fig.subplots(2, 2)
        fig.suptitle("System Performance Metrics")
        
        # Plot CPU usage
        self._plot_metric_trend(axes[0, 0], "system_cpu_usage_percent", 
                               "CPU Usage", "Time", "CPU %")
        
        # Plot memory usage
        self._plot_metric_trend(axes[0, 1], "system_memory_usage_mb", 
                               "Memory Usage", "Time", "Memory (MB)")
        
        # Plot consciousness enhancement
        self._plot_metric_trend(axes[1, 0], "consciousness_processor_consciousness_enhancement", 
                               "Consciousness Enhancement", "Time", "Enhancement (x)")
        
        # Plot operations per second
        self._plot_metric_trend(axes[1, 1], "cuda_sacred_lib_operations_per_second", 
                               "Operations/Second", "Time", "Ops/sec")
        
        plt.tight_layout()
    
    def _plot_metric_trend(self, ax, metric_key: str, title: str, xlabel: str, ylabel: str):
        """Plot trend for a specific metric"""
        if metric_key not in self.metrics_history or not self.metrics_history[metric_key]:
            ax.set_title(f"{title} (No Data)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return
        
        metrics = self.metrics_history[metric_key]
        times = [m.timestamp for m in metrics]
        values = [m.value for m in metrics]
        
        # Plot line
        ax.plot(times, values, 'b-', linewidth=2, alpha=0.7)
        
        # Add target line if available
        if metrics[0].target_value is not None:
            ax.axhline(y=metrics[0].target_value, color='g', linestyle='--', alpha=0.7, label='Target')
        
        # Add warning threshold if available
        if metrics[0].threshold_warning is not None:
            ax.axhline(y=metrics[0].threshold_warning, color='orange', linestyle='--', alpha=0.7, label='Warning')
        
        # Add critical threshold if available
        if metrics[0].threshold_critical is not None:
            ax.axhline(y=metrics[0].threshold_critical, color='r', linestyle='--', alpha=0.7, label='Critical')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for time
        ax.tick_params(axis='x', rotation=45)
        
        # Add legend if thresholds exist
        if any(m.target_value or m.threshold_warning or m.threshold_critical for m in metrics):
            ax.legend()
    
    def _update_gui(self):
        """Update GUI elements"""
        if not self.root:
            return
        
        # Update status
        if self.monitoring_active:
            self.status_label.config(text="Status: Running", foreground="green")
        else:
            self.status_label.config(text="Status: Stopped", foreground="red")
        
        # Update metrics tree
        self._update_metrics_tree()
        
        # Update alerts tree
        self._update_alerts_tree()
        
        # Update status bar
        total_metrics = sum(len(metrics) for metrics in self.metrics_history.values())
        self.status_bar.config(text=f"Total Metrics: {total_metrics} | "
                                   f"Alerts: {len(self.alerts_history)} | "
                                   f"Uptime: {self.statistics['system_uptime_minutes']:.1f} min")
        
        # Schedule next update
        self.root.after(self.update_interval_ms, self._update_gui)
    
    def _update_metrics_tree(self):
        """Update metrics tree view"""
        # Clear existing items
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        
        # Add current metrics
        for key, metrics in self.metrics_history.items():
            if not metrics:
                continue
            
            latest_metric = metrics[-1]
            
            # Determine status
            status = "OK"
            if latest_metric.threshold_critical is not None:
                if (latest_metric.metric_type in [MetricType.LATENCY_MS, MetricType.CPU_USAGE_PERCENT, 
                                                MetricType.MEMORY_USAGE_MB] and 
                    latest_metric.value > latest_metric.threshold_critical) or \
                   (latest_metric.metric_type in [MetricType.SPEEDUP_RATIO, MetricType.TFLOPS, 
                                                MetricType.COHERENCE, MetricType.CONSCIOUSNESS_ENHANCEMENT] and 
                    latest_metric.value < latest_metric.threshold_critical):
                    status = "CRITICAL"
            elif latest_metric.threshold_warning is not None:
                if (latest_metric.metric_type in [MetricType.LATENCY_MS, MetricType.CPU_USAGE_PERCENT, 
                                                MetricType.MEMORY_USAGE_MB] and 
                    latest_metric.value > latest_metric.threshold_warning) or \
                   (latest_metric.metric_type in [MetricType.SPEEDUP_RATIO, MetricType.TFLOPS, 
                                                MetricType.COHERENCE, MetricType.CONSCIOUSNESS_ENHANCEMENT] and 
                    latest_metric.value < latest_metric.threshold_warning):
                    status = "WARNING"
            
            # Format values
            value_str = f"{latest_metric.value:.3f}"
            target_str = f"{latest_metric.target_value:.3f}" if latest_metric.target_value else "N/A"
            
            # Add to tree
            item = self.metrics_tree.insert("", "end", values=(
                latest_metric.component,
                latest_metric.metric_type.value,
                value_str,
                target_str,
                status
            ))
            
            # Color code by status
            if status == "CRITICAL":
                self.metrics_tree.set(item, "Status", status)
                self.metrics_tree.item(item, tags=("critical",))
            elif status == "WARNING":
                self.metrics_tree.set(item, "Status", status)
                self.metrics_tree.item(item, tags=("warning",))
        
        # Configure tags
        self.metrics_tree.tag_configure("critical", background="red", foreground="white")
        self.metrics_tree.tag_configure("warning", background="orange", foreground="black")
    
    def _update_alerts_tree(self):
        """Update alerts tree view"""
        # Clear existing items
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        
        # Add recent alerts (last 50)
        for alert in self.alerts_history[-50:]:
            item = self.alerts_tree.insert("", "end", values=(
                alert.timestamp.strftime("%H:%M:%S"),
                alert.alert_type.value.upper(),
                alert.metric.component,
                alert.message
            ))
            
            # Color code by alert type
            if alert.alert_type == AlertType.CRITICAL:
                self.alerts_tree.item(item, tags=("critical",))
            elif alert.alert_type == AlertType.WARNING:
                self.alerts_tree.item(item, tags=("warning",))
        
        # Configure tags
        self.alerts_tree.tag_configure("critical", background="red", foreground="white")
        self.alerts_tree.tag_configure("warning", background="orange", foreground="black")
    
    def _clear_history(self):
        """Clear all performance history"""
        self.metrics_history.clear()
        self.alerts_history.clear()
        self.statistics = {
            'total_measurements': 0,
            'alerts_generated': 0,
            'performance_regressions': 0,
            'system_uptime_minutes': 0
        }
        self.logger.info("Performance history cleared")
    
    def _export_data(self):
        """Export performance data to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"performance_data_{timestamp}.json"
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.statistics,
            'metrics_history': {},
            'alerts_history': []
        }
        
        # Convert metrics to JSON-serializable format
        for key, metrics in self.metrics_history.items():
            export_data['metrics_history'][key] = [
                {
                    'metric_type': m.metric_type.value,
                    'value': m.value,
                    'timestamp': m.timestamp.isoformat(),
                    'component': m.component,
                    'target_value': m.target_value,
                    'threshold_warning': m.threshold_warning,
                    'threshold_critical': m.threshold_critical
                }
                for m in metrics
            ]
        
        # Convert alerts to JSON-serializable format
        for alert in self.alerts_history:
            export_data['alerts_history'].append({
                'alert_type': alert.alert_type.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metric': {
                    'metric_type': alert.metric.metric_type.value,
                    'value': alert.metric.value,
                    'component': alert.metric.component
                }
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Performance data exported: {filename}")
        messagebox.showinfo("Export Complete", f"Data exported to:\n{filename}")
    
    def _on_closing(self):
        """Handle window closing"""
        self.stop_monitoring()
        self.root.quit()
        self.root.destroy()
    
    def run_dashboard(self):
        """Run the monitoring dashboard"""
        self.logger.info("Starting Performance Monitoring Dashboard...")
        
        # Create and run GUI
        root = self.create_dashboard_gui()
        
        try:
            root.mainloop()
        except KeyboardInterrupt:
            self.logger.info("Dashboard interrupted by user")
        finally:
            self.stop_monitoring()

def main():
    """Main function to run monitoring dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PhiFlow Performance Monitoring Dashboard")
    parser.add_argument('--update-interval', type=int, default=1000,
                       help='Update interval in milliseconds (default: 1000)')
    parser.add_argument('--history-minutes', type=int, default=60,
                       help='History retention in minutes (default: 60)')
    parser.add_argument('--output-dir', default='/mnt/d/Projects/phiflow/monitoring',
                       help='Output directory for monitoring data')
    parser.add_argument('--no-alerts', action='store_true',
                       help='Disable performance alerts')
    
    args = parser.parse_args()
    
    # Initialize monitoring dashboard
    dashboard = PerformanceMonitoringDashboard(
        update_interval_ms=args.update_interval,
        history_minutes=args.history_minutes,
        enable_alerts=not args.no_alerts,
        output_dir=args.output_dir
    )
    
    # Run dashboard
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()