# Android SDK Setup Guide for Flutter

## Problem
You're getting: `[!] No Android SDK found. Try setting the ANDROID_HOME environment variable.`

This means you need to install Android SDK to build APKs.

## Solution: Install Android Studio

### Step 1: Download Android Studio

1. Visit: https://developer.android.com/studio
2. Download Android Studio for Windows
3. Run the installer (about 1GB download)

### Step 2: Install Android Studio

1. Run the installer
2. Choose "Standard" installation type
3. Accept all licenses
4. Let it download the Android SDK components (this takes 10-20 minutes)

### Step 3: Configure Flutter

After Android Studio is installed, run:

```bash
flutter doctor --android-licenses
```

Accept all the licenses by typing `y` when prompted.

### Step 4: Verify Installation

```bash
flutter doctor
```

You should see checkmarks (✓) for:
- Flutter
- Android toolchain
- Android Studio

### Step 5: Build Your APK

```bash
flutter build apk --release
```

---

## Alternative: Quick Setup with Command-Line Tools (Advanced)

If you don't want to install Android Studio (saves ~5GB), you can use command-line tools only:

### 1. Download Android Command Line Tools

Visit: https://developer.android.com/studio#command-tools
Download "Command line tools only" for Windows

### 2. Extract and Setup

```powershell
# Create Android SDK directory
New-Item -ItemType Directory -Path "C:\Android\sdk"

# Extract the downloaded zip to C:\Android\sdk\cmdline-tools\latest
```

### 3. Set Environment Variables

```powershell
# Run as Administrator
[System.Environment]::SetEnvironmentVariable('ANDROID_HOME', 'C:\Android\sdk', 'User')
[System.Environment]::SetEnvironmentVariable('Path', $env:Path + ';C:\Android\sdk\cmdline-tools\latest\bin;C:\Android\sdk\platform-tools', 'User')
```

**Important**: Restart your terminal after setting environment variables!

### 4. Install Required SDK Components

```bash
# Install SDK
sdkmanager "platform-tools" "platforms;android-33" "build-tools;33.0.0"

# Accept licenses
flutter doctor --android-licenses
```

---

## Quick Verification Steps

After installation, run these commands to verify:

```bash
# Check Flutter setup
flutter doctor

# Check Android SDK path
echo %ANDROID_HOME%

# Should show: C:\Android\sdk (or your Android Studio SDK path)
```

---

## Expected `flutter doctor` Output

After successful setup:

```
Doctor summary (to see all details, run flutter doctor -v):
[✓] Flutter (Channel stable, 3.38.9, on Microsoft Windows...)
[✓] Android toolchain - develop for Android devices (Android SDK version 33.0.0)
[✓] Chrome - develop for the web
[!] Android Studio (not installed) - Only if using command-line tools
[✓] VS Code (version 1.XX.X)
[✓] Connected device (1 available)
```

---

## Estimated Time

- **Android Studio Installation**: 30-45 minutes (including downloads)
- **Command-line Tools**: 15-20 minutes

## Storage Requirements

- **Android Studio**: ~8-10 GB
- **Command-line Tools Only**: ~3-4 GB

---

## After Installation

Once Android SDK is installed, you can build your APK:

```bash
# Navigate to your project
cd d:\SignSpeak\SignSpeak-FYP\front-end-mobile

# Build release APK
flutter build apk --release

# APK location:
# build/app/outputs/flutter-apk/app-release.apk
```

---

## Alternative: Use a Physical Device for Development

If you don't want to build an APK right now, you can:

1. Enable **Developer Options** on your Android phone
2. Enable **USB Debugging**
3. Connect phone via USB
4. Run: `flutter run` (installs debug version directly)

This way you can test the app without building a release APK, but you'll need to keep the phone connected to your laptop.
