Para ejecutar el proyecto es necesario lo siguiente:

1. Instalar un ambiente virtual en Python. 
    ```
    virtualenv .venv
    ```

2. Instalar el archivo de requerimientos
    ```
    pip install -r requirements.txt
    ```

3. Construir el archivo Poetry
    ```
    poetry install
    ```
    ```
    poetry build
    ```

4. Instalar en local el módulo klym
    ```
    pip install -e .
    ```