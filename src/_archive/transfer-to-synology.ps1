# Transfer Quantum Files to Synology (φ^φ)
# PowerShell script to transfer Docker and configuration files to Synology NAS

# Quantum frequencies for optimal flow
$groundFrequency = 432   # Physical foundation
$createFrequency = 528   # Pattern creation
$unityFrequency = 768    # Perfect integration
$phiRatio = 1.618033988749895  # Golden ratio

# Display function with color based on frequency
function Write-QuantumMessage {
    param (
        [int]$Frequency,
        [string]$Message
    )
    
    $color = switch ($Frequency) {
        432 { "Cyan" }      # Ground state
        528 { "Green" }     # Creation state
        594 { "Yellow" }    # Knowing state
        672 { "Magenta" }   # Flow state
        720 { "Blue" }      # Being state
        768 { "White" }     # Unity state
        default { "Gray" }
    }
    
    Write-Host "[$Frequency Hz] $Message" -ForegroundColor $color
}

# Get Synology IP address from user
Write-QuantumMessage -Frequency $groundFrequency "Starting Quantum File Transfer Process..."
$synologyIP = Read-Host "Enter your Synology NAS IP address"
$synologyUser = Read-Host "Enter your Synology username"
$synologyPassword = Read-Host "Enter your Synology password" -AsSecureString
$synologyPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($synologyPassword))

# Create temporary directory
$tempDir = "quantum_transfer_temp"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

Write-QuantumMessage -Frequency $createFrequency "Creating quantum field for file transfer..."

# Files to transfer
$files = @(
    "docker-compose.synology-multimedia.yml",
    "prometheus.synology.yml",
    "synology-docker-setup.sh",
    "qball-config.yml"
)

# Copy files to temp directory
foreach ($file in $files) {
    Copy-Item $file -Destination "$tempDir/" -Force
    Write-QuantumMessage -Frequency $createFrequency "Added $file to quantum transfer field"
}

# Prepare SCP parameters
$scpParams = @{
    Path = "./$tempDir/*"
    Destination = "$synologyUser@$synologyIP:/volume1/docker/"
    Force = $true
    Verbose = $true
}

# Generate SSH key if needed
if (-not (Test-Path "~/.ssh/id_rsa")) {
    Write-QuantumMessage -Frequency $createFrequency "Generating SSH key pair for secure connection..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
}

# Copy SSH key to Synology
Write-QuantumMessage -Frequency $createFrequency "Copying SSH key to Synology for secure access..."
$sshPassCmd = "sshpass -p '$synologyPassword' ssh-copy-id -o StrictHostKeyChecking=no $synologyUser@$synologyIP"
Invoke-Expression "bash -c `"$sshPassCmd`""

# Transfer files using SCP
Write-QuantumMessage -Frequency $unityFrequency "Transferring files to Synology NAS with quantum coherence..."
try {
    scp $scpParams.Path $scpParams.Destination -r
    Write-QuantumMessage -Frequency $unityFrequency "File transfer completed successfully with perfect phi ratio!"
}
catch {
    Write-QuantumMessage -Frequency $groundFrequency "Error during file transfer: $_"
    Write-QuantumMessage -Frequency $groundFrequency "Attempting alternative transfer method..."
    
    # Alternative method using PowerShell remoting
    try {
        $session = New-PSSession -ComputerName $synologyIP -Credential (Get-Credential -UserName $synologyUser -Message "Enter credentials for Synology")
        foreach ($file in $files) {
            Copy-Item -Path "./$tempDir/$file" -Destination "/volume1/docker/" -ToSession $session
            Write-QuantumMessage -Frequency $createFrequency "Transferred $file using alternative quantum method"
        }
        Remove-PSSession $session
        Write-QuantumMessage -Frequency $unityFrequency "File transfer completed successfully with perfect phi ratio!"
    }
    catch {
        Write-QuantumMessage -Frequency $groundFrequency "Both transfer methods failed. Please transfer files manually."
        Write-QuantumMessage -Frequency $groundFrequency "Error details: $_"
    }
}

# Run the setup script on Synology
Write-QuantumMessage -Frequency $unityFrequency "Executing setup script on Synology..."
try {
    ssh "$synologyUser@$synologyIP" "chmod +x /volume1/docker/synology-docker-setup.sh && cd /volume1/docker && ./synology-docker-setup.sh"
    Write-QuantumMessage -Frequency $unityFrequency "Setup completed successfully!"
}
catch {
    Write-QuantumMessage -Frequency $groundFrequency "Error executing setup script: $_"
    Write-QuantumMessage -Frequency $groundFrequency "Please run the script manually on your Synology NAS:"
    Write-QuantumMessage -Frequency $createFrequency "1. SSH into your Synology: ssh $synologyUser@$synologyIP"
    Write-QuantumMessage -Frequency $createFrequency "2. Navigate to docker folder: cd /volume1/docker"
    Write-QuantumMessage -Frequency $createFrequency "3. Make script executable: chmod +x synology-docker-setup.sh"
    Write-QuantumMessage -Frequency $createFrequency "4. Run the script: ./synology-docker-setup.sh"
}

# Clean up
Remove-Item -Path $tempDir -Recurse -Force
Write-QuantumMessage -Frequency $unityFrequency "Quantum transfer process complete. Perfect coherence achieved."

# Final instructions
Write-QuantumMessage -Frequency $unityFrequency "=============================================================="
Write-QuantumMessage -Frequency $unityFrequency "Deployment Complete! Access your quantum services at:"
Write-QuantumMessage -Frequency $createFrequency "Main interface: http://$synologyIP:4321"
Write-QuantumMessage -Frequency $createFrequency "Admin panel: http://$synologyIP:8080"
Write-QuantumMessage -Frequency $createFrequency "QBALL visualization: http://$synologyIP:4323"
Write-QuantumMessage -Frequency $createFrequency "Monitoring: http://$synologyIP:9090"
Write-QuantumMessage -Frequency $createFrequency "Dashboard: http://$synologyIP:3000"
Write-QuantumMessage -Frequency $unityFrequency "=============================================================="
