import psutil
import socket
import json
import time
from datetime import datetime
from collections import defaultdict
import subprocess
import platform


class SystemMonitor:
    def __init__(self):
        self.process_whitelist = [
            'python.exe', 'python', 'pythonw.exe',
            'cv2', 'mediapipe', 'explorer.exe',
            'dwm.exe', 'csrss.exe', 'wininit.exe',
            'services.exe', 'lsass.exe', 'svchost.exe',
            'System', 'Registry'
        ]
        self.suspicious_processes = []
        self.network_connections = []
        self.browser_processes = []

    def get_running_processes(self):
        """Get all currently running processes"""
        processes = []
        suspicious = []

        for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
            try:
                pinfo = proc.info
                process_name = pinfo['name'].lower()

                # Flag suspicious processes
                if self._is_suspicious(process_name, pinfo):
                    suspicious.append({
                        'name': pinfo['name'],
                        'pid': pinfo['pid'],
                        'exe': pinfo.get('exe', 'N/A'),
                        'timestamp': datetime.now().isoformat()
                    })

                processes.append({
                    'name': pinfo['name'],
                    'pid': pinfo['pid']
                })

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return processes, suspicious

    def _is_suspicious(self, process_name, pinfo):
        """Check if process is suspicious during exam"""
        suspicious_keywords = [
            'chrome', 'firefox', 'edge', 'safari', 'opera', 'brave',  # Browsers
            'chatgpt', 'bard', 'claude',  # AI tools
            'discord', 'slack', 'teams', 'zoom', 'skype',  # Communication
            'anydesk', 'teamviewer', 'vnc',  # Remote access
            'whatsapp', 'telegram', 'messenger',  # Messaging
            'notepad++', 'sublime', 'vscode', 'pycharm',  # Code editors
            'cmd', 'powershell', 'terminal',  # Command line (if not whitelisted)
        ]

        for keyword in suspicious_keywords:
            if keyword in process_name:
                return True

        return False

    def get_network_connections(self):
        """Monitor active network connections"""
        connections = []
        websites_accessed = set()

        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'ESTABLISHED':
                    try:
                        remote_ip = conn.raddr.ip if conn.raddr else None
                        remote_port = conn.raddr.port if conn.raddr else None

                        if remote_ip:
                            # Try to resolve hostname
                            try:
                                hostname = socket.gethostbyaddr(remote_ip)[0]
                            except:
                                hostname = remote_ip

                            # Get process using this connection
                            try:
                                proc = psutil.Process(conn.pid)
                                process_name = proc.name()
                            except:
                                process_name = "Unknown"

                            connection_info = {
                                'remote_ip': remote_ip,
                                'remote_port': remote_port,
                                'hostname': hostname,
                                'process': process_name,
                                'timestamp': datetime.now().isoformat()
                            }

                            connections.append(connection_info)

                            # Flag if it's a web connection
                            if remote_port in [80, 443, 8080, 8443]:
                                websites_accessed.add(hostname)

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

        except Exception as e:
            print(f"⚠️ Network monitoring error: {e}")

        return connections, list(websites_accessed)

    def get_browser_tabs(self):
        """Attempt to detect open browser tabs (Windows only, requires admin)"""
        browser_activity = []

        # This is limited without browser extensions
        # We can only detect browser processes, not specific tabs

        browsers = ['chrome.exe', 'firefox.exe', 'msedge.exe', 'opera.exe', 'brave.exe']

        for proc in psutil.process_iter(['name', 'pid', 'cmdline']):
            try:
                if proc.info['name'].lower() in browsers:
                    browser_activity.append({
                        'browser': proc.info['name'],
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(proc.info.get('cmdline', [])),
                        'timestamp': datetime.now().isoformat()
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return browser_activity

    def get_clipboard_content(self):
        """Monitor clipboard for suspicious activity (optional)"""
        try:
            if platform.system() == 'Windows':
                import win32clipboard
                win32clipboard.OpenClipboard()
                try:
                    clipboard_data = win32clipboard.GetClipboardData()
                    win32clipboard.CloseClipboard()
                    return clipboard_data
                except:
                    win32clipboard.CloseClipboard()
                    return None
            else:
                # For Linux/Mac, use xclip or pbpaste
                return None
        except:
            return None

    def take_snapshot(self):
        """Take a complete system snapshot"""
        processes, suspicious = self.get_running_processes()
        connections, websites = self.get_network_connections()
        browsers = self.get_browser_tabs()

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'total_processes': len(processes),
            'suspicious_processes': suspicious,
            'network_connections': connections,
            'websites_accessed': websites,
            'browser_activity': browsers,
        }

        return snapshot

    def continuous_monitor(self, interval=10, duration=None):
        """Continuously monitor system activity"""
        snapshots = []
        start_time = time.time()

        print(f"🔍 Starting system monitoring (interval: {interval}s)")

        try:
            while True:
                snapshot = self.take_snapshot()
                snapshots.append(snapshot)

                # Print alerts for suspicious activity
                if snapshot['suspicious_processes']:
                    print(f"⚠️ Suspicious processes detected: {len(snapshot['suspicious_processes'])}")
                    for proc in snapshot['suspicious_processes']:
                        print(f"   - {proc['name']} (PID: {proc['pid']})")

                if snapshot['websites_accessed']:
                    print(f"🌐 Active web connections: {', '.join(snapshot['websites_accessed'][:3])}")

                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n✅ Monitoring stopped")

        return snapshots


def save_monitoring_data(snapshots, filename="system_monitoring.json"):
    """Save monitoring data to file"""
    with open(filename, 'w') as f:
        json.dump(snapshots, f, indent=4)
    print(f"💾 Monitoring data saved to {filename}")


if __name__ == "__main__":
    # Test the monitor
    monitor = SystemMonitor()
    snapshot = monitor.take_snapshot()

    print("\n📊 Current System Snapshot:")
    print(f"Total Processes: {snapshot['total_processes']}")
    print(f"Suspicious Processes: {len(snapshot['suspicious_processes'])}")
    print(f"Active Connections: {len(snapshot['network_connections'])}")
    print(f"Websites Accessed: {snapshot['websites_accessed']}")
    print(f"Browser Activity: {len(snapshot['browser_activity'])}")
