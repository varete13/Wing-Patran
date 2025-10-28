# %%
# Librerias genericas
import numpy as np
import matplotlib.pyplot as plt
# Librerias para estructura y datos
from typing import Literal
from sortedcontainers import SortedDict
from itertools import cycle
# Librerias para la parte de CAD
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
# Librerias principalmente para aspectos esteticos
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from labellines import labelLines
# %%
class NACAProfile:
    NUMBER_POINTS = 31

    def __init__(self, 
                 name: str,
                 z_pos:float=0,
                 offset:tuple[float,float]=(0,0),
                 angle:float=0,
                 chord:float=1,
                 distribution:Literal['Lineal','Logarithmic']='Lineal'
                 ):        

        
        if len(name) == 4 and name.isdigit():
            self.name = name
            self.t = int(name[2:]) / 100
            self.m = int(name[0]) / 100
            self.p = int(name[1]) / 10
            self.distribution=distribution
            self.z_pos=z_pos
            self.offset=offset
            self.angle=angle
            self.chord=chord

        else:
            raise ValueError("Invalid NACA profile name. It should be a 4-digit string, e.g., '2412'.")

    def yt(self, x):
        return 5 * self.t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    def yc(self, x):
        x = np.asarray(x)
        yc = np.where(
            x < self.p,
            self.m / self.p**2 * (2 * self.p * x - x**2),
            self.m / (1 - self.p)**2 * ((1 - 2 * self.p) + 2 * self.p * x - x**2)
        )
        return yc

    def theta(self, x):
        x = np.asarray(x)
        dyc_dx = np.where(
            x < self.p,
            2 * self.m / self.p**2 * (self.p - x),
            2 * self.m / (1 - self.p)**2 * (self.p - x)
        )
        return np.arctan(dyc_dx)

    def rotate(self, x, y, angle):
        x = np.asarray(x)
        y = np.asarray(y)
        angle_rad = np.radians(-angle)
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        return x_rot, y_rot

    def contour(self):
        if self.distribution == 'Lineal':
            x = np.linspace(0, 1, self.NUMBER_POINTS)
        elif self.distribution == 'Logarithmic':
            x = np.logspace(-2, 0, self.NUMBER_POINTS)
            x[0] = 0.0  # Ensure the first point is exactly 0
        
        yt = self.yt(x)
        yc = self.yc(x)
        theta = self.theta(x)

        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Concatenate upper and lower surfaces for plotting
        x_coords = np.concatenate([xu[:-1], xl[::-1]])
        y_coords = np.concatenate([yu[:-1], yl[::-1]])

        x_rotated, y_rotated = self.rotate(x_coords, y_coords, angle=self.angle)
        
        x_coords_final = x_rotated*self.chord + self.offset[0]
        y_coords_final = y_rotated*self.chord + self.offset[1]


        return x_coords_final, y_coords_final
    
    def mean_line(self, num_points=100):
        x = [0,1]
        yc = self.yc(x)
        x_rotated, y_rotated = self.rotate(x, yc, angle=self.angle)
        x_final = x_rotated*self.chord + self.offset[0]
        y_final = y_rotated*self.chord + self.offset[1]
        return x_final, y_final

    def plot_points(self):
        x, y = self.contour()
        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, label=f'NACA {self.name}')
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi, yi, str(i), fontsize=9, ha='center', va='bottom')


        plt.title(f'NACA {self.name} Airfoil Profile')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()
# %%
class ALA:

    Points_ids=1001000
    Ribs_point_skip=1000

    Group_ribs_name='Ribs'
    Surface_ids=10010
    Ribs_surface_skip=10    

    Group_spars_name='Spars'
    Spar_ids=20010
    Ribs_spar_skip=10

    Group_stringers_name='Stringers'
    Stringer_ids=300100
    Ribs_stringer_skip=100


    def __init__(self,
        envergdura:int=20,
        profile_map:dict[float,float]={0:2442,1:2442},
        chord_map:dict[float,float]={0:6,1:0.5},
        angle_map:dict[float,float]={0:45,1:0},
        flinge_map:dict[float,float]={0:0,1:4},
        elevation_map:dict[float,float]={0:0,1:2},
        rib_locations:list[float]=np.linspace(0,1,5).tolist(),
        spar_locations:list[float]=[0.4,0.7],
        stringer_locations:list[float]=[0.1,0.2,0.3,0.5,0.6,0.8,0.9],
        map_fits:Literal['Linear','Polynomic']='Linear'
    ):
        
        self.envergdura=envergdura
        self.profile_map=SortedDict(profile_map)
        self.chord_map=SortedDict(chord_map)
        self.angle_map=SortedDict(angle_map)
        self.flinge_map=SortedDict(flinge_map)
        self.elevation_map=SortedDict(elevation_map)
        self.rib_locations=rib_locations
        self.spar_locations=spar_locations
        self.stringer_locations=stringer_locations
        self.map_fits=map_fits

        self._stetic_setup()
    
    def _stetic_setup(self):
        self._color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        self._profile_colors={str(entry):next(self._color_cycle) for entry in set(list(self.profile_map.values()))}

        self._lighten_color= lambda color, amount=0.5: tuple(
            np.array(mcolors.to_rgb(color)) + (1 - np.array(mcolors.to_rgb(color))) * amount
            )

    @staticmethod
    def _current_pick(dic,numero):
        i = dic.bisect_right(numero)
        if i == 0:
            return None
        return dic.peekitem(i - 1)[1]
    
    def profile_transitions(self):
        # profiles_dict=SortedDict(self.profile_map)


        perfiles_NACA=[]

        for profile_loc in self.profile_map.keys():

            # Interpolate parameters
            profile_name=self._current_pick(self.profile_map,profile_loc)

            if self.map_fits=='Linear':

                chord=np.interp(profile_loc,list(self.chord_map.keys()),list(self.chord_map.values()))
                angle=np.interp(profile_loc,list(self.angle_map.keys()),list(self.angle_map.values()))
                flinge=np.interp(profile_loc,list(self.flinge_map.keys()),list(self.flinge_map.values()))
                elevation=np.interp(profile_loc,list(self.elevation_map.keys()),list(self.elevation_map.values()))
            
            offset_x=flinge
            offset_y=elevation


            profile=NACAProfile(str(int(profile_name)),
                                offset=(offset_x,offset_y),
                                z_pos=profile_loc*self.envergdura,
                                angle=angle,
                                chord=chord,
                                )
            
            perfiles_NACA.append(profile)
        
        return perfiles_NACA
    
    def plot_profile_transition(self):
            perfiles=ala.profile_transitions()

            plt.figure(figsize=(10,5))
            for n,perfil in enumerate(perfiles):
                x,y=perfil.contour()
                plt.plot(x,y,lw=3,ls='--',color=self._profile_colors[perfil.name],alpha=0.7,zorder=-n)
                plt.fill(x,y,color=self._lighten_color(self._profile_colors[perfil.name],1-0.2),zorder=-n)
                xm,ym=perfil.mean_line()

                plt.plot(xm,ym,'--',color='gray')
            
            plt.axis('equal')
            plt.title('NACA Airfoil Profiles along the Wing')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')

            plt.legend(handles=[Patch(color=c, label=f'NACA {l}') for l, c in self._profile_colors.items()],title='Profiles')
            
            plt.grid()
            plt.show()

    def profile_ribs(self):
        rib_positions_list=[]

        rib_profiles=[]

        for rib_loc in self.rib_locations:
            
            profile_name=self._current_pick(self.profile_map,rib_loc)

            chord=np.interp(rib_loc,list(self.chord_map.keys()),list(self.chord_map.values()))
            angle=np.interp(rib_loc,list(self.angle_map.keys()),list(self.angle_map.values()))
            flinge=np.interp(rib_loc,list(self.flinge_map.keys()),list(self.flinge_map.values()))
            elevation=np.interp(rib_loc,list(self.elevation_map.keys()),list(self.elevation_map.values()))

            offset_x=flinge
            offset_y=elevation


            profile=NACAProfile(str(int(profile_name)),
                                offset=(offset_x,offset_y),
                                z_pos=rib_loc*self.envergdura,
                                angle=angle,
                                chord=chord,
                                )

            rib_profiles.append(profile)
        return rib_profiles

    def plot_rib_positions_yz(self):
        perfiles=ala.profile_ribs()

        plt.figure(figsize=(10,5))

        for n,perfil in enumerate(perfiles):
            x,y=perfil.contour()

            plt.plot(x,y,label=f'{perfil.z_pos:.1f} m',lw=3,ls='--',color=self._profile_colors[perfil.name],alpha=0.7,zorder=-n)
            plt.fill(x,y,color=self._lighten_color(self._profile_colors[perfil.name],1-0.2),zorder=-n)

        xvals=[0.5 * (perfil.offset[0] * 2 + perfil.chord) for perfil in perfiles]

        labelLines(plt.gca().get_lines(),xvals=xvals ,zorder=40)

        plt.axis('equal')
        plt.title('Ribs along the Wing')
        plt.xlabel('Y Coordinate')
        plt.ylabel('Z Coordinate')

        plt.legend(handles=[Patch(color=c, label=f'NACA {l}') for l, c in self._profile_colors.items()],title='Profiles')
        
        
        plt.grid()
        plt.show()

    def export_solid_step(self):
        
        def make_3d_section(profile):
            """
            profile: objeto que tiene la función contour()
            z: altura de la sección
            """
            x, y = profile.contour()
            z=profile.z_pos
            z_array = np.full_like(x, z)
            points = np.stack([x, y, z_array], axis=1)
            return points
        
        def make_wire_from_points(points):
            poly = BRepBuilderAPI_MakePolygon()
            for p in points:
                poly.Add(gp_Pnt(*p))
            poly.Close()
            return poly.Wire()

        
        wires = []
        for profile in self.profile_ribs():
            points3d = make_3d_section(profile)
            wire = make_wire_from_points(points3d)
            wires.append(wire)

        # Construir el sólido
        builder = BRepOffsetAPI_ThruSections(True, True, 1e-4)
        for wire in wires:
            builder.AddWire(wire)
        builder.Build()
        solid = builder.Shape()

        stl_writer = StlAPI_Writer()
        stl_writer.Write(solid, "ala_maciza.stl")

        # STEP
        step_writer = STEPControl_Writer()
        step_writer.Transfer(solid, STEPControl_AsIs)
        step_writer.Write("ala_maciza.step")

    def export_nastran_hollow(self):
        with open('ala_hueca.ses.01','w') as file:
            
            sub_ids_largeros=[np.argmin(np.abs(np.linspace(0,1,NACAProfile.NUMBER_POINTS)-c_loc)) for c_loc in self.spar_locations]
            sub_ids_largerillos=[np.argmin(np.abs(np.linspace(0,1,NACAProfile.NUMBER_POINTS)-c_loc)) for c_loc in self.stringer_locations]
            

            #crear grupos 
            file.write(f'ga_group_create( "{self.Group_ribs_name}" )\n')
            file.write(f'ga_group_create( "{self.Group_spars_name}" )\n')
            file.write(f'ga_group_create( "{self.Group_stringers_name}" )\n')
            # file.write(f'ga_group_create( "{self.grupo_pieles}" )\n')

            #crear puntos
            file.write('# NASTRAN SES file for hollow wing structure\n')
            file.write('STRING asm_create_grid_xyz_created_ids[VIRTUAL]\n')
            
            for n_rib,perfil in enumerate(self.profile_ribs()):
                point_origin=self.Points_ids + int(n_rib*self.Ribs_point_skip)
                for n_id,(x,y) in enumerate(zip(*perfil.contour())):
                    if n_id==NACAProfile.NUMBER_POINTS * 2 -2:
                        continue  #evitar duplicar el punto final

                    point_id=point_origin + n_id
                    file.write(f'asm_const_grid_xyz("{point_id:<7.0f}","[{x:<7.4f} {y:<7.4f} {perfil.z_pos:<7.4f}]", "Coord 0", asm_create_grid_xyz_created_ids) \n')

                #determinar contornos de las superficies, despues crear estas y eliminar las lineas de contorno
                file.write('STRING asm_create_line_pwl_created_ids[VIRTUAL]\n')

                file.write(f'asm_const_line_pwl( "1", "Point {point_origin:.0f}:{point_origin + sub_ids_largeros[0]:.0f} {point_id - sub_ids_largeros[0]}:{point_id:.0f} {point_origin:.0f}", asm_create_line_pwl_created_ids )\n')
                file.write('STRING sgm_surface_trimmed__created_id[VIRTUAL]\n')
                file.write(f'sgm_create_surface_trimmed_v1( "{self.Surface_ids + int(n_rib*self.Ribs_surface_skip)}", "Curve 1:{sub_ids_largeros[0] * 2 + 2}", "", "", TRUE, TRUE, TRUE,TRUE, sgm_surface_trimmed__created_id )\n')
                
                file.write(f'asm_const_line_pwl( "1", "Point {point_origin + sub_ids_largeros[0]:.0f}:{point_origin + sub_ids_largeros[1]:.0f} {point_id - sub_ids_largeros[1]}:{point_id - sub_ids_largeros[0]:.0f} {point_origin + sub_ids_largeros[0]:.0f}", asm_create_line_pwl_created_ids )\n')
                file.write('STRING sgm_surface_trimmed__created_id[VIRTUAL]\n')
                file.write(f'sgm_create_surface_trimmed_v1( "{self.Surface_ids + int(n_rib*self.Ribs_surface_skip) + 1}", "Curve 1:{np.diff(sub_ids_largeros)[0] * 2 + 2}", "", "", TRUE, TRUE, TRUE,TRUE, sgm_surface_trimmed__created_id )\n')

                file.write(f'asm_const_line_pwl( "1", "Point {point_origin + sub_ids_largeros[1]:.0f}:{point_id - sub_ids_largeros[1]:.0f}  {point_origin + sub_ids_largeros[1]:.0f}", asm_create_line_pwl_created_ids )\n')
                file.write('STRING sgm_surface_trimmed__created_id[VIRTUAL]\n')
                file.write(f'sgm_create_surface_trimmed_v1( "{self.Surface_ids + int(n_rib*self.Ribs_surface_skip) + 2}", "Curve 1:{NACAProfile.NUMBER_POINTS *2 -sub_ids_largeros[1] * 2 - 2}", "", "", TRUE, TRUE, TRUE,TRUE, sgm_surface_trimmed__created_id )\n')

            # añadir costillas al grupo
            for i_spar in range(len(sub_ids_largeros)+1):
                file.write(f'ga_group_entity_add( "{self.Group_ribs_name}","Surface {self.Surface_ids + i_spar}:{self.Surface_ids + i_spar + self.Ribs_surface_skip * ( len(self.profile_ribs()) -1 )}:{self.Ribs_surface_skip}" )\n')




            #crear superficies de los largeros entre las costillas
            file.write('STRING sgm_create_surface__created_ids[VIRTUAL]\n')
            for n_rib in range(len(self.profile_ribs()) - 1):

                for i_spar in range(len(sub_ids_largeros)):
                    point_origin_i=self.Points_ids + int(n_rib*self.Ribs_point_skip) + int(sub_ids_largeros[i_spar])
                    point_end_i=point_origin_i + NACAProfile.NUMBER_POINTS *2 - sub_ids_largeros[i_spar] * 2 - 3

                    file.write(f'sgm_const_surface_vertex( "{self.Spar_ids + int(n_rib*self.Ribs_spar_skip) + i_spar}", "Point {point_origin_i}", "Point {point_origin_i + self.Ribs_point_skip}", "Point {point_end_i + self.Ribs_point_skip}", "Point {point_end_i}",sgm_create_surface__created_ids )\n')
            # añadir largeros al grupo
            for i_spar in range(len(sub_ids_largeros)):
                file.write(f'ga_group_entity_add( "{self.Group_spars_name}","Surface {self.Spar_ids + i_spar}:{self.Spar_ids + i_spar + self.Ribs_spar_skip * ( len(self.profile_ribs()) -2 )}" )\n')



            #crear perfiles largerillos
            file.write('STRING asm_create_line_pwl_created_ids[VIRTUAL]\n')
            for n_stringer,i_stringer in enumerate(sub_ids_largerillos):

                point_origin=self.Points_ids + i_stringer
                point_end=self.Points_ids + (NACAProfile.NUMBER_POINTS * 2 -2) - i_stringer
                #largerillos superiores
                file.write(f'asm_const_line_pwl( "{self.Stringer_ids + int(n_stringer * self.Ribs_stringer_skip)}", "Point {point_origin:.0f}:{point_origin + (len(self.profile_ribs()) - 1)*self.Ribs_point_skip :.0f}:{self.Ribs_point_skip}", asm_create_line_pwl_created_ids )\n')
                #largerillos inferiores
                file.write(f'asm_const_line_pwl( "{self.Stringer_ids + (2 * len(sub_ids_largerillos) - n_stringer) * self.Ribs_stringer_skip}", "Point {point_end:.0f}:{point_end +(len(self.profile_ribs()) - 1)*self.Ribs_point_skip :.0f}:{self.Ribs_point_skip}", asm_create_line_pwl_created_ids )\n')    
            
            # añadir largerillos al grupo
            for n_stringer in range(len(sub_ids_largerillos)*2):
                file.write(f'ga_group_entity_add( "{self.Group_stringers_name}","Line {self.Stringer_ids + n_stringer * self.Ribs_stringer_skip}:{self.Stringer_ids + n_stringer * self.Ribs_stringer_skip + self.Ribs_stringer_skip * ( len(self.profile_ribs()) -1 )}" )\n')

if __name__ == "__main__":
    ala=ALA(
        envergdura=20,
        profile_map={0:2412,0.5:2812,1:2412},
        chord_map={0:6,1:0.5},
        angle_map={0:16,1:-2},
        flinge_map={0:0,0.25:1,1:4},
        elevation_map={0:0,1:2},
        rib_locations=np.linspace(0,1,13).tolist(),
        spar_locations=[0.2,0.7],
        stringer_locations=[0.025,0.1,0.15,0.3,0.5,0.6,0.75,0.8,0.85,0.9],
        map_fits='Linear'
    )
    # ala.plot_profile_transition()
    # ala.plot_rib_positions_yz()
    # ala.export_solid_step()
    ala.export_nastran_hollow()




