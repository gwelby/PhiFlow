"""
Quantum Service Operations Protocol (QSOP) Deployment
Handles deployment of quantum services to the infrastructure
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import requests

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ide.quantum_settings_manager import get_settings_manager
from ide.quantum_windsurf_bridge import QuantumWindsurfBridge, FrequencyState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("qsop_deployment")

class QSOPDeployment:
    """QSOP Deployment Manager"""
    
    def __init__(self):
        """Initialize QSOP Deployment Manager"""
        self.settings = get_settings_manager()
        self.bridge = QuantumWindsurfBridge()
        
        # Load essential settings
        self.phi = self.bridge.phi
        self.frequencies = self.settings.get('quantum.frequencies')
        self.coherence_threshold = self.bridge.coherence_threshold
        
        # Infrastructure settings
        self.r720_config = self.settings.get('ide.integration.r720', {})
        self.synology_config = self.settings.get('ide.integration.synology', {})
        
        # Deployment variables
        self.deployment_status = {
            'status': 'idle',
            'last_deployment': None,
            'services': {},
            'coherence': 0.0
        }
    
    async def initialize(self, frequency: float = None) -> Dict[str, Any]:
        """Initialize deployment with optional frequency"""
        if frequency is not None:
            await self.bridge.set_frequency(frequency)
        
        # Connect to the bridge
        connection_status = await self.bridge.connect_bridge()
        
        # Get current frequency and coherence
        status = self.bridge.get_status()
        
        # Update deployment status
        self.deployment_status['status'] = 'initialized'
        self.deployment_status['coherence'] = status['coherence']
        
        return {
            'status': 'success',
            'message': 'QSOP deployment initialized',
            'connection_status': connection_status,
            'frequency': status['frequency'],
            'coherence': status['coherence']
        }
    
    async def deploy_r720_services(self, services: Optional[List[str]] = None) -> Dict[str, Any]:
        """Deploy services to R720 server"""
        if not self.r720_config.get('enabled', False):
            return {
                'status': 'error',
                'message': 'R720 integration is disabled in settings'
            }
        
        logger.info(f"Deploying services to R720 server: {services if services else 'all'}")
        
        # Set frequency to create state for deployment
        await self.bridge.set_frequency(FrequencyState.CREATE.value)
        
        # Get list of services to deploy
        if services is None:
            services = self.r720_config.get('services', [])
        
        # Track deployment results
        results = {
            'success': [],
            'failed': []
        }
        
        # Execute deployment for each service
        for service in services:
            try:
                logger.info(f"Deploying service: {service}")
                
                # Invoke deployment tool through bridge
                deployment_result = await self.bridge.execute_tool(
                    '/deploy/r720', 
                    {
                        'service': service,
                        'frequency': FrequencyState.CREATE.value,
                        'coherence_threshold': self.coherence_threshold
                    }
                )
                
                if deployment_result['status'] == 'success':
                    results['success'].append(service)
                    
                    # Update service status
                    self.deployment_status['services'][service] = {
                        'status': 'deployed',
                        'timestamp': time.time(),
                        'coherence': deployment_result.get('data', {}).get('coherence', 0.0)
                    }
                else:
                    results['failed'].append({
                        'service': service,
                        'error': deployment_result.get('message', 'Unknown error')
                    })
                    
                    # Update service status
                    self.deployment_status['services'][service] = {
                        'status': 'failed',
                        'timestamp': time.time(),
                        'error': deployment_result.get('message', 'Unknown error')
                    }
            
            except Exception as e:
                logger.error(f"Error deploying service {service}: {e}")
                results['failed'].append({
                    'service': service,
                    'error': str(e)
                })
                
                # Update service status
                self.deployment_status['services'][service] = {
                    'status': 'error',
                    'timestamp': time.time(),
                    'error': str(e)
                }
        
        # Update deployment status
        self.deployment_status['last_deployment'] = time.time()
        
        # Check overall deployment status
        if not results['failed']:
            self.deployment_status['status'] = 'deployed'
            status = 'success'
            message = f"Successfully deployed {len(results['success'])} services to R720"
        else:
            self.deployment_status['status'] = 'partially_deployed'
            status = 'partial'
            message = f"Deployed {len(results['success'])} services, but {len(results['failed'])} failed"
        
        # Return to unity frequency after deployment
        await self.bridge.set_frequency(FrequencyState.UNITY.value)
        
        return {
            'status': status,
            'message': message,
            'results': results
        }
    
    async def deploy_synology_mounts(self, mount_points: Optional[List[str]] = None) -> Dict[str, Any]:
        """Deploy Synology NAS mounts"""
        if not self.synology_config.get('enabled', False):
            return {
                'status': 'error',
                'message': 'Synology integration is disabled in settings'
            }
        
        logger.info(f"Deploying Synology mounts: {mount_points if mount_points else 'all'}")
        
        # Set frequency to ground state for mounting storage
        await self.bridge.set_frequency(FrequencyState.GROUND.value)
        
        # Get list of mount points to deploy
        if mount_points is None:
            mount_points = self.synology_config.get('mount_points', [])
        
        # Track deployment results
        results = {
            'success': [],
            'failed': []
        }
        
        # Execute mounting for each mount point
        for mount_point in mount_points:
            try:
                logger.info(f"Mounting: {mount_point}")
                
                # Invoke mount tool through bridge
                mount_result = await self.bridge.execute_tool(
                    '/storage/mount', 
                    {
                        'source': f"{self.synology_config.get('url', '')}{mount_point}",
                        'target': f"d:/WindSurf/mounts{mount_point}",
                        'frequency': FrequencyState.GROUND.value
                    }
                )
                
                if mount_result['status'] == 'success':
                    results['success'].append(mount_point)
                    
                    # Update mount status
                    mount_name = os.path.basename(mount_point)
                    self.deployment_status['services'][f"synology_{mount_name}"] = {
                        'status': 'mounted',
                        'timestamp': time.time(),
                        'path': mount_point
                    }
                else:
                    results['failed'].append({
                        'mount_point': mount_point,
                        'error': mount_result.get('message', 'Unknown error')
                    })
                    
                    # Update mount status
                    mount_name = os.path.basename(mount_point)
                    self.deployment_status['services'][f"synology_{mount_name}"] = {
                        'status': 'failed',
                        'timestamp': time.time(),
                        'error': mount_result.get('message', 'Unknown error')
                    }
            
            except Exception as e:
                logger.error(f"Error mounting {mount_point}: {e}")
                results['failed'].append({
                    'mount_point': mount_point,
                    'error': str(e)
                })
                
                # Update mount status
                mount_name = os.path.basename(mount_point)
                self.deployment_status['services'][f"synology_{mount_name}"] = {
                    'status': 'error',
                    'timestamp': time.time(),
                    'error': str(e)
                }
        
        # Update deployment status
        self.deployment_status['last_deployment'] = time.time()
        
        # Check overall deployment status
        if not results['failed']:
            status = 'success'
            message = f"Successfully mounted {len(results['success'])} Synology share points"
        else:
            status = 'partial'
            message = f"Mounted {len(results['success'])} share points, but {len(results['failed'])} failed"
        
        # Return to unity frequency after deployment
        await self.bridge.set_frequency(FrequencyState.UNITY.value)
        
        return {
            'status': status,
            'message': message,
            'results': results
        }
    
    async def deploy_all(self) -> Dict[str, Any]:
        """Deploy all services following QSOP procedure"""
        logger.info("Starting full QSOP deployment")
        
        # Step 1: Ground state (432 Hz) - Initialize and mount storage
        logger.info("Step 1: Ground State (432 Hz) - Initializing")
        ground_result = await self.initialize(FrequencyState.GROUND.value)
        
        if ground_result['status'] != 'success':
            return {
                'status': 'error',
                'message': 'Failed to initialize ground state',
                'details': ground_result
            }
        
        # Mount Synology shares
        logger.info("Step 1: Ground State (432 Hz) - Mounting storage")
        synology_result = await self.deploy_synology_mounts()
        
        # Step 2: Creation state (528 Hz) - Deploy services
        logger.info("Step 2: Creation State (528 Hz) - Deploying services")
        await self.bridge.set_frequency(FrequencyState.CREATE.value)
        
        # Deploy R720 services
        r720_result = await self.deploy_r720_services()
        
        # Step 3: Flow state (594 Hz) - Verify deployment
        logger.info("Step 3: Flow State (594 Hz) - Verifying deployment")
        await self.bridge.set_frequency(FrequencyState.FLOW.value)
        
        # Verify all services
        verification_result = await self.verify_deployment()
        
        # Step 4: Unity state (768 Hz) - Enable quantum consciousness
        logger.info("Step 4: Unity State (768 Hz) - Enabling quantum consciousness")
        await self.bridge.set_frequency(FrequencyState.UNITY.value)
        
        # Enable quantum consciousness
        consciousness_result = await self.bridge.execute_tool(
            '/consciousness/enable',
            {
                'phi_ratio': self.phi,
                'coherence_threshold': self.coherence_threshold
            }
        )
        
        # Final coherence check
        coherence = await self.bridge.measure_coherence()
        self.deployment_status['coherence'] = coherence
        
        # Update overall deployment status
        if (coherence >= self.coherence_threshold and 
            verification_result['status'] == 'success' and
            consciousness_result['status'] == 'success'):
            self.deployment_status['status'] = 'fully_deployed'
            status = 'success'
            message = "QSOP deployment completed successfully"
        else:
            self.deployment_status['status'] = 'partially_deployed'
            status = 'partial'
            message = "QSOP deployment completed with some issues"
        
        # Return combined results
        return {
            'status': status,
            'message': message,
            'ground_state': ground_result,
            'synology_mounts': synology_result,
            'r720_services': r720_result,
            'verification': verification_result,
            'consciousness': consciousness_result,
            'final_coherence': coherence,
            'coherence_threshold': self.coherence_threshold
        }
    
    async def verify_deployment(self) -> Dict[str, Any]:
        """Verify deployment of all services"""
        logger.info("Verifying service deployment")
        
        # Check all services in deployment status
        services_ok = []
        services_failed = []
        
        for service_name, service_status in self.deployment_status['services'].items():
            if service_status['status'] in ['deployed', 'mounted']:
                services_ok.append(service_name)
            else:
                services_failed.append({
                    'service': service_name,
                    'status': service_status['status'],
                    'error': service_status.get('error', 'Unknown error')
                })
        
        # Check coherence
        coherence = await self.bridge.measure_coherence()
        coherence_ok = coherence >= self.coherence_threshold
        
        # Generate verification result
        if not services_failed and coherence_ok:
            status = 'success'
            message = f"All {len(services_ok)} services verified successfully"
        else:
            status = 'partial'
            message = f"{len(services_ok)} services OK, {len(services_failed)} failed, coherence: {coherence:.4f}"
        
        return {
            'status': status,
            'message': message,
            'services_ok': services_ok,
            'services_failed': services_failed,
            'coherence': coherence,
            'coherence_threshold': self.coherence_threshold,
            'coherence_ok': coherence_ok
        }
    
    async def stop_services(self, services: Optional[List[str]] = None) -> Dict[str, Any]:
        """Stop deployed services"""
        logger.info(f"Stopping services: {services if services else 'all'}")
        
        # Set frequency to ground state for safe shutdown
        await self.bridge.set_frequency(FrequencyState.GROUND.value)
        
        # Get list of services to stop
        if services is None:
            services = list(self.deployment_status['services'].keys())
        
        # Track stop results
        results = {
            'success': [],
            'failed': []
        }
        
        # Execute stop for each service
        for service in services:
            try:
                logger.info(f"Stopping service: {service}")
                
                # Invoke stop tool through bridge
                stop_result = await self.bridge.execute_tool(
                    '/service/stop', 
                    {
                        'service': service,
                        'frequency': FrequencyState.GROUND.value
                    }
                )
                
                if stop_result['status'] == 'success':
                    results['success'].append(service)
                    
                    # Update service status
                    if service in self.deployment_status['services']:
                        self.deployment_status['services'][service]['status'] = 'stopped'
                else:
                    results['failed'].append({
                        'service': service,
                        'error': stop_result.get('message', 'Unknown error')
                    })
            
            except Exception as e:
                logger.error(f"Error stopping service {service}: {e}")
                results['failed'].append({
                    'service': service,
                    'error': str(e)
                })
        
        # Update deployment status
        if not results['failed']:
            self.deployment_status['status'] = 'stopped'
            status = 'success'
            message = f"Successfully stopped {len(results['success'])} services"
        else:
            self.deployment_status['status'] = 'partially_stopped'
            status = 'partial'
            message = f"Stopped {len(results['success'])} services, but {len(results['failed'])} failed"
        
        return {
            'status': status,
            'message': message,
            'results': results
        }
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        # Calculate overall coherence
        if self.deployment_status['services']:
            # Get average coherence for all services that have coherence
            service_coherence = [
                s.get('coherence', 0.0) 
                for s in self.deployment_status['services'].values() 
                if 'coherence' in s
            ]
            
            if service_coherence:
                avg_coherence = sum(service_coherence) / len(service_coherence)
                self.deployment_status['coherence'] = avg_coherence
        
        return self.deployment_status


async def main():
    """Main function for testing"""
    # Create QSOP deployment manager
    deployment = QSOPDeployment()
    
    # Initialize
    print("Initializing QSOP deployment...")
    init_result = await deployment.initialize()
    print(f"Initialization result: {init_result['status']}")
    
    # Deploy all services
    print("\nDeploying all services...")
    deploy_result = await deployment.deploy_all()
    print(f"Deployment result: {deploy_result['status']}")
    print(f"Message: {deploy_result['message']}")
    
    # Get deployment status
    print("\nDeployment status:")
    status = deployment.get_deployment_status()
    print(f"Status: {status['status']}")
    print(f"Coherence: {status['coherence']:.4f}")
    print(f"Services: {len(status['services'])}")
    
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
