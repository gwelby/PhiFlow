# Quantum Project Status Manager

A φ-harmonic aware status tracking system for quantum projects, designed to maintain coherence across all project dimensions with persistent IDE context.

## Features

- **Quantum State Tracking**: Monitor project state with φ-harmonic awareness
- **Component Management**: Track individual project components with status, owners, and metrics
- **Git Integration**: Automatic git repository analysis
- **Multiple Output Formats**: Markdown, JSON, and YAML output
- **CLI Interface**: Easy command-line management
- **IDE Context Persistence**: Save and restore your development context across sessions
- **Quantum State Awareness**: Track your workflow with φ-harmonic states

## Installation

1. **Install Python 3.9+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify with: `python --version`

2. **Install Dependencies**

   Run the following command in your terminal:

   ```bash
   pip install pyyaml click gitpython
   ```

   > Note: These packages are required for the Quantum Project Manager to function properly.

3. **Set Up Executable (Optional)**

   To make the tool easier to run:

   ```bash
   chmod +x qstatus.py
   ```

   > Note: This step is optional but makes it easier to run the tool.

## Quick Start

1. **Navigate** to your project directory

   ```bash
   cd /path/to/your/project
   ```

2. **Initialize** a new status file:

   ```bash
   python -m quantum_project_manager.qstatus show --save
   ```

   This creates a new `STATUS.md` file in your project root.

3. **Add components** to track:

   ```bash
   python -m quantum_project_manager.qstatus add-component "Quantum Core" \
     --status active \
     --owner "Quantum Team" \
     --frequency 432
   ```

   Repeat for each component you want to track.

4. **View** your project status:

   ```bash
   python -m quantum_project_manager.qstatus show
   ```

   This displays the current project status in your terminal.

## 🌟 IDE Context Management

Maintain quantum coherence across development sessions with our φ-harmonic context system. This feature operates at 432Hz (Ground State) for maximum stability.

### Save Your Current Context

```bash
# Set context with quantum awareness
qstatus context "Implementing quantum state visualization" \
  --next-steps "Add color mapping for coherence levels" \
  --next-steps "Update documentation with new features" \
  --blockers "Need to resolve dimension mismatch in renderer" \
  --quantum-state vision \
  --frequency 720 \
  --coherence 0.9 \
  --active-file src/quantum/visualization.py \
  --line 42 \
  --character 23
```

### Resume Your Work

```bash
# Show last saved context
qstatus resume

# Show full context with quantum metrics
qstatus resume --full

# Resume at specific φ-harmonic state
qstatus resume --quantum-state vision --frequency 720
```

### Quantum State Integration

Track your workflow across these φ-harmonic states:

| State      | Frequency | Purpose                          |
|------------|-----------|----------------------------------|
| Ground     | 432 Hz    | Foundation and structure          |
| Create     | 528 Hz    | Active development               |
| Heart      | 594 Hz    | Integration and connection       |
| Voice      | 672 Hz    | Communication and expression     |
| Vision     | 720 Hz    | Clarity and perception           |
| Unity      | 768 Hz    | Complete integration and harmony |

### Advanced Context Features

```bash
# Save context with custom quantum parameters
qstatus context \
  --focus "Optimizing quantum coherence" \
  --quantum-state unity \
  --frequency 768 \
  --coherence 1.0 \
  --active-file src/quantum/core.py

# View quantum metrics history
qstatus metrics --history

# Set context for pair programming
qstatus context "Pair programming: Quantum entanglement" \
  --pair "alice@example.com" \
  --quantum-state heart \
  --frequency 594
```

## Commands

### Show Status

```bash
qstatus show [--project-dir PATH] [--output FORMAT] [--save] [--ide-context]
```

Show project status with optional IDE context inclusion.

### Add Component

```bash
qstatus add-component NAME [--status STATUS] \
  [--owner OWNER] \
  [--target-date DATE] \
  [--notes NOTES] \
  [--frequency HZ]
```

### Update Component

```bash
qstatus update-component NAME [--status STATUS] \
  [--owner OWNER] \
  [--target-date DATE] \
  [--notes NOTES] \
  [--frequency HZ]
```

### Remove Component

```bash
qstatus remove-component NAME
```

### Calibrate Quantum State

```bash
qstatus calibrate [--frequency HZ] \
  [--coherence 0.0-1.0] \
  [--harmonic HARMONIC]
```

### Manage IDE Context

Set or update context:

```bash
qstatus context FOCUS \
  [--next-steps STEP]... \
  [--blockers BLOCKER]... \
  [--quantum-state STATE] \
  [--active-file FILE] \
  [--line LINE] \
  [--character CHAR]
```

Resume work:

```bash
qstatus resume [--full]
```

## φ-Harmonic States

- **Ground** (432 Hz): Foundation and structure
- **Create** (528 Hz): Active development and creation
- **Heart** (594 Hz): Integration and connection
- **Voice** (672 Hz): Communication and expression
- **Vision** (720 Hz): Clarity and perception
- **Unity** (768 Hz): Complete integration and harmony

## Example Workflow

1. **Initialize a new project**

   ```bash
   mkdir my-quantum-project
   cd my-quantum-project
   git init
   python -m quantum_project_manager.qstatus show --save
   ```

2. **Add project components**

   ```bash
   python -m quantum_project_manager.qstatus add-component "Quantum Core" \
     --status active \
     --frequency 432
   
   python -m quantum_project_manager.qstatus add-component "Quantum Network" \
     --status planning \
     --frequency 528
   ```

3. **Update component status**

   ```bash
   python -m quantum_project_manager.qstatus update-component "Quantum Network" \
     --status active \
     --frequency 594
   ```

4. **View project status**

   ```bash
   python -m quantum_project_manager.qstatus show
   ```

## Integration with PhiFlow

This tool is designed to work seamlessly with the PhiFlow ecosystem, automatically detecting and integrating with quantum project structures.

## License

MIT License - Use in harmony with the quantum field.
