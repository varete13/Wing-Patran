import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    import plotly.graph_objects as go
    from collections import deque,defaultdict
    from itertools import pairwise
    return defaultdict, deque, go, mo, pairwise, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#Secciones de Trabajo""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Introduzca las seccione a emplear""")
    return


@app.cell
def _(mo):
    # Caja de texto
    entrada_NACA = mo.ui.text(label="Perfil NACA (4 o 5 dígitos):")

    # Lista donde guardaremos los valores válidos
    lista_perfiles_NACA = []

    def check_entry():
        valor = entrada_NACA.value.strip()
        # Verificar que sean solo números y que tenga 4 o 5 dígitos
        if valor.isdigit() and len(valor) in (4, 5):
            return True
        else:
            return False
    # Botón para confirmar
    boton_confirmacion_NACA = mo.ui.run_button(label="Agregar")

    # Mostrar los elementos en la interfaz
    mo.hstack([entrada_NACA, boton_confirmacion_NACA])
    return (
        boton_confirmacion_NACA,
        check_entry,
        entrada_NACA,
        lista_perfiles_NACA,
    )


@app.cell
def _(boton_confirmacion_NACA, check_entry, entrada_NACA, lista_perfiles_NACA):
    if boton_confirmacion_NACA.value:
        if check_entry():
            entry=entrada_NACA.value
            if entry not in lista_perfiles_NACA:
                lista_perfiles_NACA.append(entrada_NACA.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Alternativas Parametricas""")
    return


@app.cell
def _(mo):
    texto_subida_archivo = mo.md("Selecciona un archivo para subir:")
    file_coord = mo.ui.file(filetypes=[".txt", ".csv"], multiple=True)
    mo.hstack([texto_subida_archivo, file_coord])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Disposición Ala""")
    return


@app.cell
def _(lista_perfiles_NACA, mo):
    entry_b=mo.ui.text(placeholder="b  (m)")
    entry_chord=mo.ui.text(placeholder="c  (m)")
    entry_elevacion=mo.ui.text(placeholder="e  (m)")
    entry_angulo=mo.ui.text(placeholder="a  (º)")
    entry_flecha=mo.ui.text(placeholder="f  (m)")
    entry_perfil=mo.ui.dropdown(options=lista_perfiles_NACA,full_width=True)
    return (
        entry_angulo,
        entry_b,
        entry_chord,
        entry_elevacion,
        entry_flecha,
        entry_perfil,
    )


@app.cell
def _(
    add_to_global,
    entry_angulo,
    entry_b,
    entry_chord,
    entry_elevacion,
    entry_flecha,
    entry_perfil,
    mo,
):
    general_entry_2=mo.md("""
    ### Distribución de Parametros


    |b|{b}|
    |-|-|
    |cuerda|{chord}|
    |elevacion|{elevacion}|
    |angulo|{angulo}|
    |flecha|{flecha}|
    |perfil|{perfil}|
    """)

    form_2=general_entry_2.batch(b=entry_b,chord=entry_chord,elevacion=entry_elevacion,angulo=entry_angulo,flecha=entry_flecha,perfil=entry_perfil).form(clear_on_submit=True,on_change=add_to_global)
    form_2
    return


@app.cell
def _():
    DISTRIBUCION_ALA=list()
    return (DISTRIBUCION_ALA,)


@app.cell
def _(DISTRIBUCION_ALA):
    def add_to_global(cas):

        def little_filter(k,v):
            if v=='':
                return 0

            if k=='perfil':
                return str(v)
            else:
                return float(v)

        cas_filtered={key:little_filter(key,value) for key,value in cas.items()}
        DISTRIBUCION_ALA.append(cas_filtered)
    return (add_to_global,)


@app.cell
def _(DISTRIBUCION_ALA, pd):
    asta=pd.DataFrame(DISTRIBUCION_ALA)
    return (asta,)


@app.cell
def _():
    from NACAp import NACAProfile
    return (NACAProfile,)


@app.cell
def _(NACAProfile, asta, defaultdict, deque, go, mo, pairwise):

    data_preview=list()
    data_plain_preview=defaultdict(deque)

    for (index_i,datos_i),(index_i2,datos_i2) in pairwise(asta.iterrows()):

        Ala_i=NACAProfile(datos_i.loc['perfil'],chord=datos_i.loc['chord'],
                          angle=datos_i.loc['angulo'],
                          offset=(datos_i.loc['flecha'],datos_i.loc['elevacion']))

        Ala_i2=NACAProfile(datos_i2.loc['perfil'],chord=datos_i2.loc['chord'],
                          angle=datos_i2.loc['angulo'],
                          offset=(datos_i2.loc['flecha'],datos_i2.loc['elevacion']))
        x=list()
        y=list()
        z=list()

        x.extend([datos_i.loc['b']]*2 + [datos_i2.loc['b']]*2 )
        y.extend(list(Ala_i.mean_line()[0]) + list(Ala_i2.mean_line()[0][::-1]))
        z.extend(list(Ala_i.mean_line()[1]) + list(Ala_i2.mean_line()[1][::-1]))

        i = [0, 0]
        j = [1, 2]
        k = [2, 3]

        data_preview.append(
            go.Mesh3d(
                x=x,y=y,z=z,
                i=i,j=j,k=k,
                opacity=0.4,
                color='orange',
                name='Plain Drawn'
            )
        )

    for index,datos in asta.iterrows():
        Ala_i=NACAProfile(datos.loc['perfil'],chord=datos.loc['chord'],
                          angle=datos.loc['angulo'],
                          offset=(datos.loc['flecha'],datos.loc['elevacion']))

        y,z=Ala_i.contour()
        x=[datos.iloc[0]]*len(z)

        n=len(z)
        i = [0]*(n-2)              # siempre el vértice 0
        j = list(range(1, n-1))    # vértices consecutivos
        k = list(range(2, n))

        data_preview.append( 
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=1,
                name=f'NACA {datos.loc['perfil']}',
                color='turquoise'
                )
        )

    fig=go.Figure(data=data_preview)
    fig.update_layout(
        scene=dict(
            xaxis_title='b',
            yaxis_title='c',
            zaxis_title='',
            zaxis=dict(showticklabels=False),
            aspectmode='data'),
            title="Preview Ala",
    )
    mo.ui.plotly(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
