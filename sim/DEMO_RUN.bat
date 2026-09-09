@echo off
REM NRDI live demo launcher -- opens every surface of the simulation in its own real tool.
REM Run from anywhere; paths are resolved relative to this file.
setlocal
set ROOT=%~dp0..
set PY=C:\Users\Shlok\nrdi-env\Scripts\python.exe
set BLENDER="C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

echo [1/4] Dashboard (offline copy of the digital twin) in your default browser
start "" "%ROOT%\sim\dashboard\nrdi_twin.html"

echo [2/4] Finite-element mesh in the gmsh GUI (705k tets, 14 regions)
start "gmsh" %PY% -c "import gmsh,os; gmsh.initialize(); gmsh.open(r'%ROOT%\sim\out\periorbital.msh'); gmsh.option.setNumber('Mesh.SurfaceFaces',1); gmsh.option.setNumber('Mesh.VolumeEdges',0); gmsh.fltk.run(); gmsh.finalize()"

echo [3/4] Temperature field + anatomy + sensors in the browser (Plotly; the VTK viewer is blocked by Smart App Control on this PC)
if not exist "%ROOT%\sim\render\out\view_fields.html" python "%ROOT%\sim\render\view_fields_web.py" --no-open
start "" "%ROOT%\sim\render\out\view_fields.html"
REM VTK route, only if Smart App Control is off:  start "pyvista" %PY% "%ROOT%\sim\render\view_fields.py"

echo [4/4] Exploded assembly in Blender (press Space in the timeline to play)
start "blender" %BLENDER% "%ROOT%\sim\render\out\explode.blend"

echo.
echo Also available in Blender: hero_face.blend, hero_product.blend, physics_thermography.blend
echo Solver outputs to show on request: sim\out\thermal3d_full.vtu (ParaView/PyVista), sim\out\*.json
endlocal
