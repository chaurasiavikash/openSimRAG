import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Creating a new OpenSim vector database...")

# Define key OpenSim information
opensim_docs = [
    {
        "title": "OpenSim Overview",
        "content": """
OpenSim is an open source software system that lets users develop models of musculoskeletal structures and create dynamic simulations of movement. OpenSim provides a platform on which the biomechanics community can build a library of simulations that can be exchanged, tested, analyzed, and improved through a multi-institutional collaboration.

OpenSim's main features include:
- Creating and editing musculoskeletal models 
- Importing motion capture data for analysis
- Performing inverse kinematics, inverse dynamics, and forward dynamic simulations
- Analyzing muscle forces, moments, and joint reactions
- Visualizing models and simulation results in a 3D environment
- Extensible through plugins and Python/MATLAB interfaces

OpenSim is developed and maintained by the National Center for Simulation in Rehabilitation Research (NCSRR) at Stanford University. It is used by researchers in biomechanics, neuroscience, rehabilitation, orthopedics, and many other fields.
"""
    },
    {
        "title": "OpenSim Installation Guide",
        "content": """
Installing OpenSim

OpenSim is available for Windows, Mac, and Linux systems. The installation process varies by platform.

For Windows:
1. Download the OpenSim installer from https://simtk.org/frs/?group_id=91
2. Run the installer executable (OpenSim-X.X-win64.exe)
3. Follow the installation wizard instructions
4. The default installation directory is C:\\OpenSim X.X

For Mac:
1. Download the OpenSim installer .pkg file from https://simtk.org/frs/?group_id=91
2. Run the .pkg file and follow the installation instructions
3. OpenSim will be installed in your Applications folder

For Linux:
Linux users need to build OpenSim from source:
1. Clone the OpenSim repository: git clone https://github.com/opensim-org/opensim-core.git
2. Follow the build instructions in the INSTALL.md file

System Requirements:
- Windows 10/11 or macOS 10.15+ or Linux (Ubuntu 20.04+ recommended)
- 8GB RAM minimum (16GB recommended)
- 3D graphics card with OpenGL support
- 2GB free disk space
"""
    },
    {
        "title": "OpenSim Models",
        "content": """
OpenSim Models

OpenSim models are mathematical representations of musculoskeletal systems. They consist of rigid bodies (bones), joints, muscles, and other force-generating elements.

Key components of OpenSim models:
1. Bodies: Rigid segments representing bones or groups of bones
2. Joints: Connections between bodies that define allowable motion
3. Muscles: Force-generating actuators with properties like optimal fiber length and maximum isometric force
4. Constraints: Additional restrictions on model movement
5. Contact geometry: Shapes used for contact forces between bodies
6. Markers: Virtual markers that correspond to experimental markers placed on subjects

Models in OpenSim are stored in .osim files, which are XML-based text files. These files can be created and edited through the OpenSim GUI or programmatically using the OpenSim API.

Standard models available in OpenSim include:
- Gait2392: A full-body model with 23 degrees of freedom and 92 muscle-tendon actuators
- Arm26: A simple arm model with 2 degrees of freedom and 6 muscles
- RajagopalFullBody: A detailed full-body model with improved muscle paths and properties

To load a model in OpenSim:
1. Open the OpenSim application
2. Go to File > Open Model
3. Select the .osim file from your computer
4. The model will appear in the Visualizer window
"""
    },
    {
        "title": "OpenSim Markers",
        "content": """
Markers in OpenSim

Markers in OpenSim are virtual reference points placed on a model that correspond to experimental markers placed on a subject during motion capture. They are essential for scaling models to match subject-specific dimensions and for performing inverse kinematics to determine joint angles from motion capture data.

Types of markers in OpenSim:
1. Experimental markers: Physical markers placed on subjects during data collection
2. Virtual markers: Reference points placed on the OpenSim model

Working with markers in OpenSim:
1. Adding markers to a model:
   - In the OpenSim GUI, right-click on a body in the Navigator
   - Select "Add" > "Marker"
   - Specify the location of the marker in the body's reference frame
   - Give the marker a name that matches the corresponding experimental marker

2. Importing marker data:
   - Marker trajectories are typically stored in .trc (Track Row Column) files
   - Use File > Import > Motion Data to import .trc files into OpenSim

3. Marker placement for common protocols:
   - For gait analysis: Markers are typically placed on anatomical landmarks like the anterior and posterior superior iliac spines, lateral knee joint line, lateral malleolus, and metatarsal heads
   - For upper extremity analysis: Markers are placed on the acromion process, lateral epicondyle, medial epicondyle, and wrist joint

4. Using markers for scaling:
   - In the Scale Tool, markers are used to determine the dimensions of each body segment
   - Pairs of markers define the length of body segments
   - Virtual marker placement should match experimental marker placement for accurate scaling

5. Using markers for inverse kinematics:
   - In the Inverse Kinematics Tool, OpenSim minimizes the distance between experimental and virtual markers
   - Each marker can be assigned a different weighting based on confidence in its placement and tracking
   - Marker weights typically range from 1-20, with higher values forcing closer tracking

6. Common marker errors:
   - Marker placement inconsistencies between the model and the subject
   - Soft tissue artifacts causing marker movement relative to underlying bones
   - Marker dropout or occlusion during motion capture
   - Mislabeled markers in the motion capture system

To visualize markers in OpenSim:
1. Enable "Show markers" in the Navigator panel
2. Adjust marker appearance (size, color) in the Properties window
3. Use the "Show only" option to isolate specific markers for better visualization
"""
    },
    {
        "title": "OpenSim Inverse Kinematics",
        "content": """
Inverse Kinematics in OpenSim

Inverse Kinematics (IK) is a process in OpenSim that determines joint angles based on experimental marker positions. The IK tool minimizes the weighted sum of squared distances between experimental markers and corresponding virtual markers on the model.

Steps to perform Inverse Kinematics in OpenSim:

1. Prepare your model and data:
   - Ensure your model has virtual markers that correspond to experimental markers
   - Import your marker data (.trc file) into OpenSim

2. Set up the IK Tool:
   - Go to Tools > Inverse Kinematics
   - Select your model from the dropdown menu
   - Specify the marker data file (.trc)
   - Set the time range for analysis
   - Choose which coordinates to solve for (typically all)
   - Set weights for markers (higher weights force closer tracking)
   - Specify the output motion file name (.mot)

3. Run the IK Tool:
   - Click "Run" to execute the inverse kinematics solution
   - OpenSim will compute joint angles that best match experimental markers
   - Results are saved as a motion (.mot) file

4. Assess the results:
   - Review marker errors in the Messages window
   - Load the motion file to visualize the resulting movement
   - Check for physiologically unrealistic joint angles
   - Examine the RMS marker error for each frame

Common IK parameters:
- Marker weights: Typically range from 1-20, with higher values for more reliable markers
- Constraint weight: Default is 20, controls how strictly joint constraints are enforced
- Accuracy: Default is 1e-5, determines convergence criteria

Tips for successful IK:
- Ensure virtual markers on the model match experimental marker placements
- Use higher weights for markers on bony landmarks (less soft tissue artifact)
- Use lower weights for markers prone to movement or tracking errors
- Check that all necessary degrees of freedom are unlocked in the model
- For problematic frames, try running IK on smaller time ranges

The output motion file contains:
- Time column
- One column for each coordinate (joint angle) in the model
- Values typically in radians for rotational coordinates and meters for translational coordinates
"""
    },
    {
        "title": "OpenSim Forward Dynamics",
        "content": """
Forward Dynamics in OpenSim

Forward dynamics simulation in OpenSim predicts how a model will move based on applied forces and initial conditions. Unlike inverse dynamics, which calculates forces from motion, forward dynamics calculates motion from forces.

Key concepts in forward dynamics:
1. Equations of motion: Based on Newton's laws, these differential equations describe how the model accelerates in response to forces
2. Numerical integration: The process of solving these equations over time to determine position and velocity
3. Actuators: Components that generate forces in the model (muscles, ideal force generators, etc.)
4. Controls: Input signals that specify actuator activation over time
5. Initial conditions: Starting values for coordinates (positions) and speeds (velocities)

Steps to perform a forward dynamics simulation in OpenSim:

1. Prepare your model:
   - Ensure all necessary actuators are defined
   - Set appropriate model parameters (mass, inertia, joint properties)
   - Define contact forces if applicable

2. Set up the Forward Dynamics Tool:
   - Go to Tools > Forward Dynamics
   - Select your model
   - Specify initial states (from a .sto file or the current model state)
   - Set the simulation time range
   - Define control signals (constant values or from a controls file)
   - Specify the integration parameters (step size, accuracy)
   - Set output file names

3. Run the simulation:
   - Click "Run" to execute the forward dynamics simulation
   - OpenSim will integrate the equations of motion
   - Results are saved as states (.sto) files

4. Analyze the results:
   - Load the states file to visualize the resulting movement
   - Plot variables of interest (joint angles, muscle forces, etc.)
   - Compare with experimental data if available

Common forward dynamics parameters:
- Integration step size: Typically 0.01 seconds or smaller
- Integration accuracy: Default is 1e-5
- Maximum number of steps: Limits computation time for diverging simulations

Advanced forward dynamics applications:
1. Computed Muscle Control (CMC): Uses forward dynamics with feedback to track a desired motion
2. Predictive simulations: Optimizes control signals to achieve objectives like minimal energy use
3. Monte Carlo simulations: Runs multiple forward simulations with varied parameters to analyze sensitivity

To simplify forward dynamics problems:
- Start with simple models and short time periods
- Use ideal force actuators before trying muscle-driven simulations
- Ensure model stability by checking for unrealistic joint ranges or contact forces
"""
    },
    {
        "title": "OpenSim Python API",
        "content": """
OpenSim Python API

The OpenSim Python API allows users to access OpenSim functionality programmatically using Python. This enables automation, customization, and integration with other scientific computing tools.

Setting up the OpenSim Python API:
1. The Python bindings are automatically installed with OpenSim 4.0 and later
2. On Windows, use the Start menu shortcut "OpenSim Python Shell"
3. On Mac/Linux, ensure the OpenSim libraries are in your path
4. Import OpenSim in Python using: `import opensim`

Basic usage examples:

1. Loading a model:
```python
import opensim as osim

# Load a model from an .osim file
model = osim.Model("path/to/model.osim")

# Print model information
print(f"Model name: {model.getName()}")
print(f"Number of bodies: {model.getBodySet().getSize()}")
print(f"Number of muscles: {model.getMuscles().getSize()}")
```

2. Accessing model components:
```python
# Get a specific muscle
muscle = model.getMuscles().get("biceps")
print(f"Max isometric force: {muscle.getMaxIsometricForce()}")

# Get a joint
joint = model.getJointSet().get("elbow")
print(f"Joint type: {joint.getConcreteClassName()}")

# Get a coordinate (joint angle)
coord = model.getCoordinateSet().get("elbow_flexion")
print(f"Range of motion: {coord.getRangeMin()} to {coord.getRangeMax()} radians")
```

3. Running a simulation:
```python
# Create a simulation manager
state = model.initSystem()
manager = osim.Manager(model)
manager.setInitialTime(0)
manager.setFinalTime(1.0)
manager.integrate(state)
```

4. Analyzing results:
```python
# Create an analyzer
analyzer = osim.MuscleAnalysis(model)
analyzer.setStartTime(0)
analyzer.setEndTime(1.0)
analyzer.printResults("results")
```

Common Python API tasks:
- Model creation and modification
- Batch processing of simulations
- Custom analyses and visualizations
- Parameter optimization
- Integration with machine learning workflows

The OpenSim Python API provides access to the same functionality as the C++ API but with the convenience and flexibility of Python. It's particularly useful for research workflows that require custom analyses or batch processing.

For detailed documentation, refer to the OpenSim API reference at: https://simtk.org/api_docs/opensim/api_docs/
"""
    },
    {
        "title": "OpenSim File Formats",
        "content": """
OpenSim File Formats

OpenSim uses several file formats to store models, motions, forces, and other data. Understanding these formats is essential for effective use of OpenSim.

1. .osim - OpenSim Model Files
   - XML-based format that defines the musculoskeletal model
   - Contains definitions of bodies, joints, muscles, contact geometry, etc.
   - Can be viewed and edited with any text editor
   - Example: <Body name="tibia_r"> defines the right tibia body segment

2. .mot - Motion Files
   - Tab or space-delimited text files containing time series data
   - Typically store joint angles over time (kinematics)
   - First line contains number of rows
   - Second line contains column labels
   - First column is usually time
   - Example: Used for storing inverse kinematics results

3. .sto - Storage Files
   - Similar to .mot files but with additional header information
   - Store any time series data (states, accelerations, forces, etc.)
   - More general than .mot files and used for a wider range of data
   - Example: Used for storing forward dynamics simulation results

4. .trc - Track Row Column Files
   - Store 3D marker trajectory data from motion capture
   - Each marker has X, Y, Z columns
   - Include header with marker names and metadata
   - Example: Used as input for the Scale and IK tools

5. .xml - Tool Setup Files
   - Store settings for OpenSim tools (Scale, IK, ID, CMC, etc.)
   - Allow saving and reusing tool configurations
   - Example: IKTool settings saved as IKTool.xml

6. .vtp - Visualization Toolkit Polygon Files
   - Store 3D geometry for visualization
   - Used for muscle paths, bone meshes, contact geometry, etc.
   - Binary format, not human-readable
   - Example: Bone meshes used for model visualization

7. .opensim - OpenSim Document Files
   - Container for multiple files related to a single project
   - Includes models, motions, forces, settings, etc.
   - Actually a zip file that can be renamed and extracted
   - Example: Used to share complete projects between collaborators

Tips for working with OpenSim files:
- Always check file headers to understand the content structure
- Use consistent naming conventions for related files
- Backup original files before modification
- When editing XML files manually, maintain proper XML syntax
- For large datasets, use programming interfaces (Python, MATLAB) for batch processing

Common issues with file formats:
- Column count mismatch in .mot or .sto files
- Missing end-of-line characters in text files
- XML syntax errors in .osim files
- Incorrect marker names between .trc and .osim files
- Unit inconsistencies between files
"""
    }
]

# Define the output directory for the vector database
# Use the same directory as in your script
output_dir = "./chroma_db"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize the embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Process each document
print("Processing documents and creating chunks...")
all_chunks = []

for i, doc_info in enumerate(opensim_docs):
    # Create a Document object
    document = Document(
        page_content=doc_info["content"],
        metadata={
            "title": doc_info["title"],
            "source": f"Manual entry: {doc_info['title']}",
            "section": "",
            "content_type": "documentation"
        }
    )
    
    # Split the document into chunks
    chunks = text_splitter.split_documents([document])
    
    print(f"Created {len(chunks)} chunks from '{doc_info['title']}'")
    all_chunks.extend(chunks)

# Create a new vector database
print(f"Creating vector database with {len(all_chunks)} chunks...")
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=output_dir,
    collection_name="opensim_docs"
)

# Persist the database
vectorstore.persist()

print(f"Vector database created successfully at {output_dir}")
print(f"Total documents: {len(all_chunks)}")
print("\nYou can now use this database with the RAG system:")
print(f"python enhanced_rag.py --vector_db_path {output_dir}")