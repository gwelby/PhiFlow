import aiohttp
import json
import logging
from urllib.parse import urlencode
from typing import Optional, Dict

class SynologyAuth:
    def __init__(self, host: str = "192.168.100.32", port: int = 5001, secure: bool = True):
        self.host = host
        self.port = port
        self.secure = secure
        self.base_url = f"{'https' if secure else 'http'}://{host}:{port}/webapi"
        self.sid: Optional[str] = None
        self.session = None
        
    async def connect(self, username: str, password: str) -> bool:
        """Connect to Synology DSM WebAPI"""
        try:
            # Create aiohttp session with SSL verification disabled for local network
            self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
            
            # Get API info
            api_info = await self._get_api_info()
            
            # Authenticate
            auth_url = f"{self.base_url}/auth.cgi"
            params = {
                'api': 'SYNO.API.Auth',
                'version': '3',
                'method': 'login',
                'account': username,
                'passwd': password,
                'session': 'QuantumFileManager',
                'format': 'cookie'
            }
            
            async with self.session.get(auth_url, params=params) as response:
                data = await response.json()
                
                if data.get('success'):
                    self.sid = data['data']['sid']
                    logging.info("✨ Connected to Synology DSM")
                    return True
                else:
                    logging.error(f"Authentication failed: {data.get('error', {}).get('code')}")
                    return False
                    
        except Exception as e:
            logging.error(f"Connection failed: {str(e)}")
            return False
            
    async def disconnect(self):
        """Disconnect from Synology DSM"""
        if self.session:
            try:
                if self.sid:
                    # Logout
                    logout_url = f"{self.base_url}/auth.cgi"
                    params = {
                        'api': 'SYNO.API.Auth',
                        'version': '3',
                        'method': 'logout',
                        'session': 'QuantumFileManager'
                    }
                    
                    async with self.session.get(logout_url, params=params) as response:
                        await response.json()
                        
            except Exception as e:
                logging.error(f"Logout error: {str(e)}")
                
            finally:
                await self.session.close()
                self.sid = None
                
    async def _get_api_info(self) -> Dict:
        """Get Synology DSM API information"""
        info_url = f"{self.base_url}/query.cgi"
        params = {
            'api': 'SYNO.API.Info',
            'version': '1',
            'method': 'query',
            'query': 'SYNO.API.Auth,SYNO.FileStation'
        }
        
        async with self.session.get(info_url, params=params) as response:
            return await response.json()
            
    def get_auth_params(self) -> Dict:
        """Get authentication parameters for API calls"""
        return {
            '_sid': self.sid
        } if self.sid else {}
