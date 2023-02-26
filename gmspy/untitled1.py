import tkinter
import pyvista

from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor


# Setup for root window
root = tkinter.Tk()
root.title("pyvista tk Demo")

frame = tkinter.Frame(root)
frame.pack(fill=tkinter.BOTH, expand=1, side=tkinter.TOP)

# create an instance of a pyvista.Plotter to be used for tk
mesh = pyvista.Sphere()
pl = pyvista.Plotter()
pl.add_mesh(mesh)

# Setup for rendering window interactor
renwininteract = vtkTkRenderWindowInteractor(root, rw=pl.ren_win,
                                             width=400, height=400)
renwininteract.Initialize()
renwininteract.pack(side='top', fill='both', expand=1)
renwininteract.Start()

# Begin execution by updating the renderer and starting the tkinter
# loop
pl.render()
root.mainloop()