import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go

    from abc import ABC, abstractmethod
    return (mo,)


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
