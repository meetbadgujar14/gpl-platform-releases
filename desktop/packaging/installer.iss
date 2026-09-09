; GPL Platform — Inno Setup installer script
; Produces: GPLPlatform_Setup_v{version}.exe
;
; Usage:
;   iscc /Q /DMyAppVersion=1.0.0 desktop\packaging\installer.iss
;
; Prerequisites:
;   - Inno Setup 6 from jrsoftware.org/isdl.php
;   - PyInstaller dist already built at desktop\packaging\dist\GPL Platform\

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

#define MyAppName      "GPL Platform"
#define MyAppExeName   "GPL Platform.exe"
#define MyAppPublisher "GPL Platform"
#define MyAppMutex     "GPLPlatformSingleInstance"
; DistDir must be passed as an absolute path via /DDistDir="..." on the command line

[Setup]
AppId={{B3A7F2C1-4D8E-4A6B-9F2E-1C3D5E7A9B0C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={userpf}\{#MyAppName}
DefaultGroupName={#MyAppName}
; OutputDir set via /O flag or defaults to .iss location
OutputBaseFilename=GPLPlatform_Setup_v{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
AppMutex={#MyAppMutex}
UninstallDisplayName={#MyAppName}
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=GPL Platform Desktop App

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Everything from the PyInstaller onedir output
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";    Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\GPLPlatform\updater"
Type: filesandordirs; Name: "{localappdata}\GPLPlatform\staging"
