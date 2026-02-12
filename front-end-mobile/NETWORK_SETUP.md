# Network Setup Guide for SignSpeak

## Prerequisites for Mobile Testing

### 1. Find Your Laptop's IP Address

Open PowerShell and run:
```powershell
ipconfig
```

Look for the "Wireless LAN adapter Wi-Fi" section and note the **IPv4 Address**.
Example: `192.168.100.2`

### 2. Update Flutter App Configuration

Edit `lib/main.dart` (around line 30) and replace the IP:

```dart
create: (_) => AppState(
  backendIp: '192.168.100.2',  // Replace with YOUR laptop's IP
  backendPort: 8000,
),
```

Also update `lib/services/prediction_service.dart` (line 30):
```dart
static const String defaultIp = '192.168.100.2'; // Replace with YOUR laptop's IP
```

### 3. Configure FastAPI Backend

Ensure your backend is accessible on the network. In your FastAPI backend code:

```python
if __name__ == "__main__":
    import uvicorn
    # IMPORTANT: Use host="0.0.0.0" to accept connections from any network interface
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. Configure Windows Firewall

**Option A: Using PowerShell (Recommended)**

Run PowerShell as Administrator and execute:

```powershell
# Allow incoming connections on port 8000
New-NetFirewallRule -DisplayName "FastAPI SignSpeak" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

**Option B: Using Windows Firewall GUI**

1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Click "Inbound Rules" → "New Rule..."
4. Select "Port" → Next
5. Select "TCP" and enter port "8000" → Next
6. Select "Allow the connection" → Next
7. Check all profiles (Domain, Private, Public) → Next
8. Name it "FastAPI SignSpeak" → Finish

### 5. Verify Network Connectivity

**Step 1: Start your FastAPI backend**
```bash
python your_backend_file.py
```

**Step 2: Test from laptop browser**
```
http://localhost:8000/health
```
Should return success.

**Step 3: Find your laptop's IP and test from mobile browser**

On your mobile phone, open a browser and visit:
```
http://192.168.100.2:8000/health
```
(Replace with your actual laptop IP)

If this works, the Flutter app will also work!

### 6. Build and Install Flutter App

**Build the APK:**
```bash
flutter build apk --release
```

**Transfer the APK:**
The APK will be located at:
```
build/app/outputs/flutter-apk/app-release.apk
```

Transfer this file to your phone via:
- USB cable
- Email
- Cloud storage (Google Drive, Dropbox, etc.)
- ADB: `adb install build/app/outputs/flutter-apk/app-release.apk`

**Install on phone:**
1. Enable "Install from Unknown Sources" in Android settings
2. Open the APK file and install

### 7. Testing Checklist

Before testing the full app:

- [ ] Both laptop and phone are on the same WiFi network
- [ ] Laptop IP address is updated in the Flutter app
- [ ] FastAPI backend is running with `host="0.0.0.0"`
- [ ] Windows Firewall allows port 8000
- [ ] Mobile browser can access `http://<LAPTOP_IP>:8000/health`
- [ ] APK is built and installed on phone

### Troubleshooting

**Problem: Mobile can't connect to backend**

1. **Check if both devices are on the same network**
   ```powershell
   # On laptop, check connected network
   netsh wlan show interfaces
   ```

2. **Test connectivity with ping**
   
   From your phone, use a network tool app (like "Fing" or "Network Analyzer") to ping your laptop's IP.

3. **Temporarily disable Windows Firewall** (for testing only)
   ```powershell
   # Run as Administrator
   Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
   ```
   
   If it works after disabling firewall, you know it's a firewall issue. Re-enable it and add the proper rule:
   ```powershell
   Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True
   ```

4. **Check FastAPI is listening on all interfaces**
   ```powershell
   netstat -an | findstr :8000
   ```
   Should show `0.0.0.0:8000` or `*:8000`, NOT `127.0.0.1:8000`

5. **Check laptop's IP hasn't changed**
   
   If your laptop has DHCP, the IP can change. Consider setting a static IP or checking it each time.

**Problem: App works but predictions are slow**

- Ensure your WiFi has good signal strength
- Consider reducing throttle duration in `lib/providers/app_state.dart` (line 22)
- Check your backend processing time

**Problem: Camera permissions denied**

- Go to Android Settings → Apps → SignSpeak → Permissions
- Grant Camera permission

### Production Tips

For production deployment:

1. **Use a static IP** for your laptop or deploy backend to a cloud server
2. **Use HTTPS** instead of HTTP for secure communication
3. **Add authentication** to your FastAPI backend
4. **Implement offline mode** in the Flutter app for when network is unavailable
5. **Add network quality indicators** to warn users of poor connectivity

### Network Architecture

```
┌─────────────────┐
│  Mobile Phone   │
│  (Flutter App)  │
│  📱             │
└────────┬────────┘
         │ WiFi
         │ HTTP POST /predict
         │
    ┌────▼────────────────┐
    │   WiFi Router       │
    │   192.168.1.1       │
    └────┬────────────────┘
         │
         │
    ┌────▼────────────────┐
    │   Laptop            │
    │   192.168.100.2     │
    │   FastAPI Backend   │
    │   Port 8000         │
    └─────────────────────┘
```

Both devices must be connected to the same WiFi network for local communication.
