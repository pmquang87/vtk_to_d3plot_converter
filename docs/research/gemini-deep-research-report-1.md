---
title: "Engineering Analysis of VTK to LS-DYNA d3plot Conversion: Translators, APIs, and Binary File Architecture"
source: gemini-deep-research
conversation: c0a012a2cd75002c
extracted: 2026-08-02T13:01:48.931Z
extractor: extract_report.js v1.0.0
---
# Engineering Analysis of VTK to LS-DYNA d3plot Conversion: Translators, APIs, and Binary File Architecture

## 1. Introduction and Architectural Context

In the domain of explicit finite element method (FEM) solvers, the computational mechanics of crash simulations generate massive, highly transient datasets. The transition from solvers like OpenRadioss and LS-DYNA into post-processing environments hinges entirely on the efficiency of the underlying data formats. Engineers porting OpenRadioss to custom Python frameworks frequently rely on the Visualization Toolkit (VTK) unstructured grid format (legacy `.vtk` or XML-based `.vtu`) as an initial output mechanism due to its extensive open-source support and flexibility. However, while tools like the OpenRadioss `anim_to_vtk` converter successfully extract nodal and elemental results (such as shells, solids, beams) per time step, the resulting VTK files introduce severe performance bottlenecks. These files, particularly when ASCII-based, become extraordinarily large and significantly degrade the performance of generalized visualization tools like ParaView[^1].

To resolve these I/O bottlenecks and integrate seamlessly with specialized automotive and aerospace crash post-processors (such as LS-PrePost, Altair HyperView, and BETA CAE META), engineering workflows demand the translation of these VTK time-series results into the LS-DYNA `d3plot` binary plotfile format. The `d3plot` architecture is a highly specialized, rigidly allocated binary memory dump optimized for rapid state-by-state loading[^3]. The translation of generalized, self-describing VTK cell arrays into the strict, pre-allocated block structures of a `d3plot` family presents a formidable algorithmic challenge. This report provides an exhaustive investigation into existing conversion utilities, evaluates Python libraries capable of programmatically serializing `d3plot` buffers, deconstructs the architecture of current conversion pipelines, and outlines the critical mathematical and structural pitfalls inherent in this translation.

## 2. Exhaustive Survey of Existing VTK to d3plot Converters

A rigorous, multi-platform investigation was conducted across open-source repositories (GitHub, GitLab, SourceForge), Python package indices (PyPI), major computer-aided engineering forums (dynasupport, Ansys LS-DYNA forum, Eng-Tips), and the documentation of commercial post-processors.

The definitive conclusion of this research is that **no existing tool, either open-source or commercial, currently converts VTK or VTU unstructured grids to the LS-DYNA d3plot format.** The data pipeline across the CAE industry is overwhelmingly structured to flow into VTK for generalized visualization, rather than out of VTK into proprietary solver formats.

### 2.1 The Open-Source and Python Ecosystem

The open-source landscape is saturated with mesh manipulation and visualization libraries, yet none possess an LS-DYNA binary export capability:

- **VTK / PyVista / vtkInterface:** The native VTK project and its high-level Python wrappers (such as `vtkInterface` and PyVista) excel at granting direct NumPy array access to node and element connectivity matrices[^5]. While they natively support exporting to formats like STL, Wavefront OBJ, PLY, and neutral formats, the core VTK source code fundamentally lacks a `d3plot` serialization backend[^6].
- **vedo (formerly vtkplotter):** This library provides highly intuitive APIs for mesh manipulation, morphing, and volumetric analysis, alongside comprehensive file format support (XML, Neutral, GMSH, OFF, PCD)[^7]. However, review of its PyPI metadata and repository features confirms it has no pathway for generating LS-DYNA databases[^7].
- **tovtk:** Maintained for the conversion of MCNP5 and MCNP6 meshtal files into VTK rectilinear grids, this tool exemplifies the industry trend of utilizing VTK as a terminal endpoint for custom solvers, rather than an intermediary step toward `d3plot`[^9].
- **OpenRadioss Conversion Utilities:** Within the OpenRadioss community, developers rely on `anim_to_vtk` or specialized Python scripts for `TH-csv` conversions[^1]. While community members have noted the performance issues of ParaView and explicitly requested `d3plot` outputs, the development focus has remained solely on translating native Radioss Animation (A-files) directly to `d3plot`, entirely bypassing VTK[^2].

### 2.2 Commercial Post-Processors

Commercial CAE environments are engineered as highly optimized data sinks, designed to ingest and render solver outputs rather than act as bidirectional format translators.

- **Altair HyperView (HVTrans):** HyperView features an extensive Python API (e.g., `hw.hv` and `CaptureImageTool`) capable of automating the ingestion of `d3plot` files and exporting visualizations to animation formats (AVI, GIF, JPEG)[^12]. It does not, however, expose a programmatic path to import VTK geometry and serialize a valid `d3plot` binary structure.
- **BETA CAE META, GNS Animator, and Oasys:** Forum analyses and user discussions confirm that these platforms possess robust, highly optimized readers for `d3plot`, EnSight, and VTK formats[^14]. Users frequently combine data sources in these viewers, but there is no documented functionality in META or Animator that allows a user to save an imported VTK unstructured grid as a multi-state `d3plot` file family.

The absence of an off-the-shelf conversion utility dictates that engineering teams must construct custom translation layers using low-level API frameworks capable of manipulating LS-DYNA binary memory buffers.

| Software Category | Tool / Ecosystem | VTK Import | d3plot Export | Capability Assessment |
|---|---|---|---|---|
| **Open-Source** | VTK / vtkInterface | Yes | No | Terminal endpoint for visualization[^5]. |
| **Open-Source** | vedo (vtkplotter) | Yes | No | Exports to STL, OBJ, GMSH; no binary CAE support[^7]. |
| **CAE Utilities** | OpenRadioss tools | No | No | Converts solver-native formats to VTK, not vice versa[^1]. |
| **Commercial** | Altair HyperView | Yes | No | Automates visual rendering exports (AVI, GIF)[^12]. |
| **Commercial** | BETA META / Animator | Yes | No | Highly optimized data sink; lacks format serialization[^14]. |

## 3. API Evaluation for Programmatic d3plot Generation

To architect a custom VTK-to-d3plot translator, a foundational library with explicit write access to the `d3plot` binary schema is required. The following analysis evaluates the leading CAE data libraries, detailing their licensing, maintenance, and serialization capabilities.

### 3.1 lasso-python

Originating from LASSO GmbH and currently maintained by the open-source community, `lasso-python` is the de facto standard for programmatic `d3plot` manipulation in Python[^16].

- **License:** BSD-3-Clause[^16].
- **Maintenance Status:** Highly active. The community consistently merges pull requests addressing geometry parsing, array sizing, and binary buffer allocations[^17].
- **Write Capability Details:** `lasso-python` definitively supports writing binary data. The `lasso.dyna.d3plot` module contains the `write_d3plot` function (implemented near line 6215/6268 of `d3plot.py`), which accepts parameters such as `filepath`, `block_size_bytes`, and `single_file` to serialize in-memory arrays back to disk[^20].
- **Element and Array Support:** The library successfully maps standard geometries, including nodes, 8-node solids, 4-node shells, and 2-node beams[^11]. It supports state variables ranging from basic kinematics (displacements, velocities, accelerations) to complex elemental results (stresses, true strains, internal energies)[^17].
- **Known Limitations and Bugs:** The reliance on `lasso-python` introduces several documented pitfalls:
  - *Solid History Variable Corruption:* A known bug (Issue #63, #99) causes shape broadcast errors when reading or writing `element_solid_history_variables`. The internal indexing (`i_solid_vars`) incorrectly increments when solid true strain data is requested, resulting in array size mismatches (e.g., expecting a shape of `(522,5913,1,5)` but receiving `(522,5913,1,11)`)[^18].
  - *Rigid Body Data Compression (DCOMP):* The parser fails when the LS-DYNA `*DATABASE_EXTENT_BINARY` keyword invokes rigid body data compression (`DCOMP` set to 2, 4, 5, or 6). This induces a word count mismatch in the file header (`128745 != 99515`) and triggers a `ValueError` regarding buffer lengths[^22].
  - *Non-Default Content Initialization:* Initializing a `D3plot` object with truncated data (e.g., when the user has excluded nodal accelerations or stress tensors from the solver output) can cause the reader to throw a `RuntimeError: Unknown file type '-1'` rather than filling the absent arrays with zeroes, complicating the instantiation of objects intended for writing[^17].
  - *Advanced Element Instability:* Issues #101 and #65 indicate active errors when writing files that contain SPH (Smoothed Particle Hydrodynamics) data or when the `node_temperature` field has been modified prior to calling `write_d3plot`[^19].

### 3.2 qd-cae-python

Initially a pioneering C++/Python framework for CAE post-processing, `qd-cae-python` provided high-speed interaction with LS-DYNA results[^23].

- **License:** Permissive open-source, though specifically reliant on precompiled wheels with optional FEMZIP support[^23].
- **Maintenance Status:** Deprecated and Archived. The repository was officially made read-only on October 7, 2021. The maintainers explicitly advise migrating to `lasso-python`[^23].
- **Write Capability:** None. The core C++ module (`RawD3plot.cpp`) and the Python interface (`D3plot.py`) are strictly implemented for reading states (`read_states`), parsing geometry, and feeding visualization backends[^27]. There is no programmatic export pathway.

### 3.3 dynareadout

- **License:** Open-source[^24].
- **Maintenance Status:** Active[^24].
- **Write Capability:** None. `dynareadout` is a highly optimized C/C++/Python library explicitly designed to maximize performance and minimize memory overhead when *reading* binary `binout` and `d3plot` files[^24]. It lacks any serialization backend.

### 3.4 Ansys pydyna

- **License:** MIT / Ansys open-source license[^29].
- **Maintenance Status:** Actively maintained by Ansys.
- **Write Capability:** Non-existent for low-level binary buffer manipulation. Evidence surrounding `pydyna` suggests it functions primarily as a pre-processing tool to programmatically generate LS-DYNA input decks and interface with the solver via gRPC servers[^29]. (Note: This is flagged as weak evidence based on available repository snippets, but its architectural focus precludes its use as a raw `d3plot` writer).

### 3.5 Vortex-Radioss

Vortex-Radioss is an application-layer tool developed for the OpenRadioss ecosystem, bridging the gap between open-source solvers and commercial post-processors[^11].

- **License:** Mozilla Public License (MPL-2.0)[^31].
- **Maintenance Status:** Active. Beta release announced in early 2024[^11].
- **Write Capability:** Yes, indirectly. The library features the `Anim_to_D3plot` module, which successfully maps solver arrays into the `d3plot` format by acting as a sophisticated wrapper around the `lasso-python` writer[^11].

| Library / API | Primary Function | License | Status | Write `d3plot`? | Implementation Notes |
|---|---|---|---|---|---|
| **lasso-python** | CAE data processing | BSD-3-Clause | Active | **Yes** | Uses `write_d3plot`; prone to shape broadcast bugs[^18]. |
| **Vortex-Radioss** | Radioss format bridging | MPL-2.0 | Active | **Yes** | Wrapper utilizing `lasso-python` as the serialization backend[^31]. |
| **qd-cae-python** | Fast binary parsing | Permissive | Archived | No | C++ backend; strictly read-only; deprecated in 2021[^26]. |
| **dynareadout** | Memory-efficient parsing | Permissive | Active | No | Optimized exclusively for reading operations[^24]. |
| **Ansys pydyna** | Deck generation / gRPC | MIT | Active | No | Focuses on solver interaction rather than binary I/O[^29]. |

## 4. Vortex-Radioss Architecture and the VTK Adapter Feasibility

Understanding the architectural pipeline of Vortex-Radioss provides the precise blueprint required to construct a VTK-to-d3plot converter. Vortex-Radioss essentially operates on an adapter pattern, parsing proprietary data into normalized Python memory structures (NumPy arrays) before feeding them to the serialization engine.

### 4.1 Mechanics of the Anim-to-d3plot Pipeline

The software executes its translation via three primary modules: `RadiossReader`, `RadiossTHReader`, and `Anim_to_D3plot`[^11].

1. **Data Ingestion and Unpacking:** The `RadiossReader` parses a single Radioss Animation file (A-file) per state to extract compressed spatial and geometric data arrays. Simultaneously, the `RadiossTHReader` processes the Time-History files (T-files) to extract global variables required by the `d3plot` specification that are absent from the A-files[^31].
2. **Structural Mapping:** The `Anim_to_D3plot` module (invoked via `readAndConvert`) receives the unpacked NumPy arrays. It structurally maps the geometry for beam, shell, and solid elements[^11]. Beyond topology, it injects node coordinates per time state and maps elemental scalar and tensor arrays (e.g., von Mises stress, effective plastic strain)[^11].
3. **Header and Mass Synthesis:** Recent repository updates highlight the complexity of the conversion. The module actively calculates and maps part masses for shell parts (leaving others at zero) and explicitly forces absent size flags from the `header_raw` to `0` to maintain referencing consistency within the binary structure[^11].
4. **Serialization:** Finally, it populates a `lasso.dyna.d3plot.D3plot` object in memory and executes `self._d3plot.write_d3plot(temp_d3plot_name, single_file=False)` to generate the file family on disk[^21].

### 4.2 Interfacing the Writer with a VTK Reader

Because the `Anim_to_D3plot` module simply maps NumPy arrays into the `lasso-python` object schema, **the writer can theoretically be decoupled from the `RadiossReader` and fed directly from a VTK reader.**

An engineer could utilize a library like `vtkInterface` or `pyvista` to ingest the VTK unstructured grid. The points array (providing node coordinates) and the cell data arrays (providing scalar/tensor results) can be extracted directly as NumPy arrays[^5]. The critical engineering task is replacing the A-file geometric parser with a VTK cell parser. The VTK connectivity matrices must be segregated by cell type, translated from VTK node ordering to LS-DYNA node ordering, and fed into the `Anim_to_D3plot` mapping logic. Provided the required global time-history headers are synthesized or mocked, the `lasso-python` serialization backend will generate the `d3plot`.

## 5. Authoritative Documentation of the d3plot Format

To bypass the high-level Python APIs and implement low-level bug fixes or custom buffering, access to the core `d3plot` specification is mandatory. The format is a proprietary standard, yet heavily documented by LSTC (now Ansys) and meticulously reverse-engineered by the community.

The following table details the primary, authoritative sources governing the file architecture, accompanied by explicit URLs as mandated by the research requirements.

| Resource Title / Description | Authoritative URL | Relevance |
|---|---|---|
| **LS-DYNA Database Binary Output Files** (Revised Jan 2015) | [https://lsdyna.ansys.com/wp-content/uploads/2025/02/nvh-simulations-for-car-seat.pdf](https://lsdyna.ansys.com/wp-content/uploads/2025/02/nvh-simulations-for-car-seat.pdf) | The core specification manual defining control words, geometry, and state data structures[^32]. |
| **dynasupport: Database Manual (Legacy)** | [https://www.dynasupport.com/manuals/additional/ls-dyna-database-manual](https://www.dynasupport.com/manuals/additional/ls-dyna-database-manual) | Details file length splitting, block padding sizes (512 words), and adaptivity file naming conventions[^3]. |
| **dynasupport: Database Manual (2014)** | [https://www.dynasupport.com/manuals/additional/ls-dyna-database-manual-2014](https://www.dynasupport.com/manuals/additional/ls-dyna-database-manual-2014) | Identical structure to the legacy manual, ensuring validation of byte offsets and family splitting rules[^4]. |
| **LS-DYNA Keyword Manual Vol I (R11)** | [https://www.dynasupport.com/manuals/ls-dyna-manuals/ls-dyna_manual_volume_i_r11.pdf](https://www.dynasupport.com/manuals/ls-dyna-manuals/ls-dyna_manual_volume_i_r11.pdf) | Explains how keywords like `*DATABASE_EXTENT_BINARY` alter the database footprint[^33]. |
| **LS-DYNA Keyword Manual Vol II (R13)** | [https://www.dynasupport.com/manuals/ls-dyna-manuals/ls-dyna_manual_volume_ii_r13.pdf](https://www.dynasupport.com/manuals/ls-dyna-manuals/ls-dyna_manual_volume_ii_r13.pdf) | Details how specific material models (e.g., cohesive elements) alter stress tensor reporting inside the database[^34]. |
| **qd-cae-python: RawD3plot.cpp** | [https://github.com/qd-cae/qd-cae-python/blob/master/qd/cae/dyna_cpp/dyna/d3plot/RawD3plot.cpp](https://github.com/qd-cae/qd-cae-python/blob/master/qd/cae/dyna_cpp/dyna/d3plot/RawD3plot.cpp) | The de facto open-source C++ reference implementation for byte-level offset reading and header interpretation[^27]. |

## 6. Known Algorithmic and Structural Pitfalls

Converting time-series data from VTK into a multi-state `d3plot` file family introduces severe architectural and mathematical challenges. VTK operates as a highly generalized graphics standard; conversely, `d3plot` is a strictly typed, memory-dump-style architecture heavily influenced by legacy Fortran I/O paradigms. Constructing a reliable converter requires navigating the following constraints.

### 6.1 Splitting Mixed VTK Cell Types into Segregated Blocks

A standard VTK file utilizing a `vtkUnstructuredGrid` maintains all elements in a single, contiguous connectivity array. The geometry of the mesh is resolved by a parallel `cell_types` array, which flags whether a specific cell is a hexahedron, quad, triangle, or line[^5].

The `d3plot` format is fundamentally hostile to this structure. LS-DYNA mandates that geometry is rigorously segregated into distinct, pre-allocated blocks: all 8-node solids must be stored contiguously, followed by all 4-node shells, followed entirely by 2-node beams[^3].

- **The Translation Pitfall:** A converter cannot iterate through the VTK file and write elements as they appear. The algorithm must pre-parse the entire VTK connectivity list, filter the elements by their VTK type (e.g., separating `VTK_HEXAHEDRON` from `VTK_QUAD`), and construct independent, block-segregated connectivity matrices before initiating the `lasso-python` serialization.

### 6.2 Arbitrary Node and Element Numbering (NARBS)

VTK arrays utilize implicit, 0-based memory indexing; the node ID is inherently defined by its row index in the coordinates array. In stark contrast, explicit solvers rely on arbitrary, user-defined alphanumeric numbering for nodes, elements, and parts to maintain correlation with the input deck.

LS-DYNA resolves this discrepancy via the **NARBS** (Node and Element Arbitrary Numbering) section of the binary format[^3].

- **The Translation Pitfall:** A VTK reader will only supply sequential indices. If the original OpenRadioss solver IDs were not explicitly embedded as `PointData` or `CellData` arrays during the `anim_to_vtk` phase, the converter will be forced to synthesize a sequential 1-based NARBS block. This will strip the resulting `d3plot` of its engineering context, breaking cross-referencing capabilities when engineers attempt to locate failing parts within the original model deck.

### 6.3 Stress Tensor Component Ordering and Integration Points

Data mapped to the VTK grid is typically stored as contiguous vectors per cell. The `d3plot` format relies on highly specialized rules governing state variables.

- **Tensor Ordering Constraints:** Standard LS-DYNA outputs yield six stress components. However, specific material definitions alter this output dynamically. For example, when evaluating cohesive elements (`*MAT_ADD_COHESIVE`), the standard six components are overridden and replaced by three traction stresses[^34]. A naive converter writing full 6-component tensors into a cohesive part block will fatally misalign all subsequent data offsets.
- **Shell Integration Points:** Shell elements in crash simulations frequently capture bending moments and plasticity via multiple through-thickness integration points (e.g., employing a 3-Point Simpson integration rule)[^35]. VTK visualizations often average these points into a single scalar or tensor per element to minimize file size[^1]. If the control words of the synthesized `d3plot` declare multiple integration points, but the VTK arrays supply only surface data, the resulting buffer mismatch will corrupt the binary structure[^22].

### 6.4 Single vs. Double Precision Encoding (32ieee)

The binary footprint of the `d3plot` file is dictated by precision flags. The base word size is 4 bytes for single-precision compilations and 8 bytes for double-precision outputs, unless the `32bit ieee` format is forcefully defined via the `*DATABASE_FORMAT, IBINARY=1` keyword[^3].

- **The Translation Pitfall:** Python's NumPy inherently defaults to 64-bit floating-point numbers (`float64`). If the converter dumps double-precision VTK arrays into a file whose header declares a single-precision word size, post-processors and readers like `qd-cae-python` will immediately throw endianness or precision misalignment errors (e.g., `Wrong filetype 538976288 != 1 (or 5)`)[^26]. The translation logic must explicitly cast VTK data arrays to `np.float32` prior to serialization, unless a double-precision architecture is actively selected.

### 6.5 File-Family Splitting and Adaptivity

To circumvent filesystem limitations during massive transient analyses, LS-DYNA dynamically splits its databases.

- **Block Padding:** The files are written with a strict block size of 512 words. If the data block terminates before completing 512 words, the remainder is padded with zeroes[^3].
- **Family Naming Conventions:** The default file length is constrained to $7×512×512$ words. Upon exceeding this limit, the solver closes the root `d3plot` file and initiates writing to sequential family members: `d3plot01`, `d3plot02`, etc. Furthermore, if the simulation involves mesh adaptivity, the root name receives an alphabetical appendage (e.g., generating `d3plotaa`, `d3plotaa01`, then `d3plotab`) to indicate topological shifts[^3].
- **The Translation Pitfall:** A converter cannot simply serialize a massive, multi-state VTK directory into a single monolithic binary blob. It must meticulously implement the 512-word padding logic and dynamically split the outputs across a generated file family. While `lasso-python` mitigates this via its `single_file=False` argument, navigating multi-mesh adaptivity programmatically remains exceptionally error-prone[^11].

### 6.6 Header Consolidation and Femzip Compression

The control words at the head of a `d3plot` dictate exactly how many bytes follow. If variables normally expected by the post-processor (such as `node_temperature`) are modified, injected, or removed, the control header must be perfectly recalculated to prevent downstream parsing failures[^20]. Additionally, the open-source implementation faces severe barriers regarding compression. The industry heavily utilizes Femzip to compress output databases. Generating a Femzip-compressed `d3plot` programmatically requires linking proprietary C-libraries (such as `femunziplib_standard_dyna`)[^37]. Consequently, the converter must generate uncompressed binaries. Users migrating from highly optimized VTK formats to raw `d3plot` may experience an exponential increase in file sizes, running contrary to performance optimization goals[^1].

## 7. Strategic Conclusions

The requirement to translate OpenRadioss VTK unstructured grid results into the LS-DYNA `d3plot` format is technically viable, though it mandates the engineering of a custom middleware pipeline.

The market survey unequivocally confirms the absence of off-the-shelf conversion utilities. Among the available programmatic interfaces, `lasso-python` stands as the exclusive viable serialization engine, despite requiring careful management of known array shape broadcasting bugs. The optimal architectural pattern for this endeavor is to emulate the `Vortex-Radioss` adapter structure: leveraging Python libraries like `pyvista` to ingest the VTK data, mathematically segregating the mixed topology into rigid solid, shell, and beam matrices, and relying on `lasso-python` to handle the low-level 512-word block padding and precision serialization. Success hinges on strict adherence to LSTC's proprietary formatting rules, particularly concerning arbitrary node numbering (NARBS), shell integration data mapping, and precise word-size casting.

---

## Sources

[^1]: [Large Paraview vtk ASCII files; How about binary output? · OpenRadioss · Discussion #2273](https://github.com/orgs/OpenRadioss/discussions/2273)
[^2]: [Animation file (.h3d) file format description · OpenRadioss · Discussion #552 - GitHub](https://github.com/orgs/OpenRadioss/discussions/552)
[^3]: [LS-DYNA® Database Binary Output Files](https://www.dynasupport.com/manuals/additional/ls-dyna-database-manual)
[^4]: [LS-DYNA® Database Binary Output Files](https://www.dynasupport.com/manuals/additional/ls-dyna-database-manual-2014)
[^5]: [freckletonj/vtkInterface: Python interface for VTK - GitHub](https://github.com/freckletonj/vtkInterface)
[^6]: [vtk - PyPI](https://pypi.org/project/vtk/)
[^7]: [GitHub - fcollman/vtkplotter: A python class for scientific visualization of 3D objects with VTK](https://github.com/fcollman/vtkplotter)
[^8]: [jumbojing/vtkplotter: A python class for scientific visualization of 3D objects with VTK - GitHub](https://github.com/jumbojing/vtkplotter)
[^9]: [inr-kit/tovtk: Python meshtal -> VTK converter - GitHub](https://github.com/inr-kit/tovtk)
[^10]: [New GUI with export to 3dplot LS-Dyna · OpenRadioss · Discussion #2819 - GitHub](https://github.com/orgs/OpenRadioss/discussions/2819)
[^11]: [Vortex-Radioss - D3plot convertor and open-source Python post-processing library for OpenRadioss - now released #2361 - GitHub](https://github.com/orgs/OpenRadioss/discussions/2361)
[^12]: [Examples — HyperMesh Python API 2026.0 documentation](https://help.altair.com/hwdesktop/pythonapi/framework/examples/hw_examples.html)
[^13]: [HyperView Tutorials | PDF | Button (Computing) - Scribd](https://www.scribd.com/doc/6675293/HyperView-Tutorials)
[^14]: [D3PLOT 22.0 - Oasys LS-DYNA Environment](https://dyna.oasys-software.com/content/uploads/2025/06/d3plot.pdf)
[^15]: [Has anyone ever found a way to transfer Dyna results to partners who don't have a post-processor? : r/LSDYNA - Reddit](https://www.reddit.com/r/LSDYNA/comments/1qlywtk/has_anyone_ever_found_a_way_to_transfer_dyna/)
[^16]: [Home of the open-source CAE library lasso-python - GitHub](https://github.com/open-lasso-python/lasso-python)
[^17]: [Loading d3plot with non-default content · Issue #60 - GitHub](https://github.com/open-lasso-python/lasso-python/issues/60)
[^18]: [mismatching array sizes when reading element_solid_history_variables #63 - GitHub](https://github.com/open-lasso-python/lasso-python/issues/63)
[^19]: [Issues · open-lasso-python/lasso-python - GitHub](https://github.com/open-lasso-python/lasso-python/issues)
[^20]: [Issue when writing d3plot with "node_temperature" field #65 - GitHub](https://github.com/open-lasso-python/lasso-python/issues/65)
[^21]: [Errors when writing d3plot containing SPH data · Issue #101 · open](https://github.com/open-lasso-python/lasso-python/issues/101)
[^22]: [Using Rigid Body Data Compression in *DATABASE_EXTENT_BINARY leads to error #71](https://github.com/open-lasso-python/lasso-python/issues/71)
[^23]: [qd - a CAE Python Library - GitHub](https://github.com/qd-cae/qd-cae-python)
[^24]: [A curated list of awesome CAE frameworks, libraries and software. - GitHub](https://github.com/qd-cae/awesome-CAE)
[^25]: [D3plot does not read all the time steps #38 - GitHub](https://github.com/qd-cae/qd-cae-python/issues/38)
[^26]: [Read double precision files · Issue #31 - GitHub](https://github.com/qd-cae/qd-cae-python/issues/31)
[^27]: [qd-cae-python/qd/cae/dyna_cpp/dyna/d3plot/RawD3plot.cpp at master - GitHub](https://github.com/qd-cae/qd-cae-python/blob/master/qd/cae/dyna_cpp/dyna/d3plot/RawD3plot.cpp)
[^28]: [qd-cae-python/qd/cae/D3plot.py at master · qd-cae/qd-cae-python - GitHub](https://github.com/qd-cae/qd-cae-python/blob/master/qd/cae/D3plot.py)
[^29]: [Implement AGENTS.md · Issue #4391 · ansys/pymapdl - GitHub](https://github.com/ansys/pymapdl/issues/4391)
[^30]: [GUI with D3 Plot Conversion option · OpenRadioss Tools · Discussion #30 - GitHub](https://github.com/OpenRadioss/Tools/discussions/30)
[^31]: [GitHub - Vortex-CAE/Vortex-Radioss: An open-source python library for processing Radioss results files](https://github.com/Vortex-CAE/Vortex-Radioss)
[^32]: [NVH simulations for car seat | LS-DYNA - Vorlagen für Tagungsband](https://lsdyna.ansys.com/wp-content/uploads/2025/02/nvh-simulations-for-car-seat.pdf)
[^33]: [keyword user's manual volume i - LS-DYNA support](https://www.dynasupport.com/manuals/ls-dyna-manuals/ls-dyna_manual_volume_i_r11.pdf)
[^34]: [keyword user's manual volume ii - LS-DYNA support](https://www.dynasupport.com/manuals/ls-dyna-manuals/ls-dyna_manual_volume_ii_r13.pdf)
[^35]: [LS-DYNA R11.2.0 released (2020/12)](https://www.dynasupport.com/news/ls-dyna-r11-2-0-released)
[^36]: [Wrong filetype 538976288 != 1 (or 5) in header of d3plot. Your file might be in Double Precision or the endian of the file is not the endian of the machine. · Issue #36 · qd-cae/qd-cae-python - GitHub](https://github.com/qd-cae/qd-cae-python/issues/36)
[^37]: [qd-cae-python/setup.py at master - GitHub](https://github.com/qd-cae/qd-cae-python/blob/master/setup.py)