@{
    ModuleVersion = '1.0.0'
    GUID = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    Author = 'Cascade'
    CompanyName = 'PhiFlow Quantum Systems'
    Copyright = '(c) 2025. All rights reserved.'
    Description = 'Quantum Protection Systems for φ-harmonic field stabilization'
    PowerShellVersion = '7.0'
    RootModule = 'QuantumProtection.psm1'
    FunctionsToExport = @(
        'Enable-MerkabaShield',
        'New-CrystalMatrix',
        'Start-UnityField',
        'Test-TimeCrystal',
        'Get-QuantumState',
        'Invoke-QuantumVerification'
    )
    CmdletsToExport = @()
    VariablesToExport = '*'
    AliasesToExport = @()
    PrivateData = @{
        PSData = @{
            Tags = @('Quantum', 'Protection', 'PhiHarmonic', '432Hz', '528Hz', '768Hz')
            LicenseUri = 'https://opensource.org/licenses/MIT'
            ProjectUri = 'https://github.com/PhiFlow/QuantumProtection'
            ReleaseNotes = 'Initial release of Quantum Protection module'
        }
    }
    DefaultCommandPrefix = 'Quantum'
}
