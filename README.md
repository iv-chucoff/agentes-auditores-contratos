# Agentes Auditores de Contratos

## Problemática

En contextos legales y administrativos, los contratos suelen modificarse mediante adendas o versiones actualizadas. Identificar con precisión qué cambió entre la versión original y la modificada es un proceso tedioso, propenso a errores humanos y costoso en tiempo.

Este proyecto resuelve ese problema: permite cargar imágenes de un contrato original y su versión modificada (o adenda), transcribe automáticamente su contenido y utiliza agentes de inteligencia artificial para analizar y auditar los cambios entre ambas versiones, entregando un reporte estructurado en formato JSON.

---

## Flujo del sistema

```
Imágenes del contrato (original + modificado)
        ↓
  Transcripción
        ↓
  Agente Contextualizador → mapa estructural de ambos contratos
        ↓
  Agente Auditor → detección de cambios
        ↓
  Output JSON validado con Pydantic
```

---

## Output

El resultado es un archivo `.json` con la siguiente estructura:

```text
{
  "sections_changed": [
    "PRECIO Y PAGO",
    "DEPÓSITO",
    "SEGURO CONTRA INCENDIOS",
    "PROHIBICIONES"
  ],
  "topics_touched": [
    "Condiciones de pago",
    "Depósito de garantía",
    "Obligación de seguro",
    "Restricciones de uso"
  ],
  "summary_of_the_change": "ADICIONES: Se añade la cláusula 8, 'SEGURO CONTRA INCENDIOS', que establece la obligación de la locataria de contratar un seguro contra incendios dentro de los 30 días hábiles.

ELIMINACIONES: Se elimina la cláusula 9, 'PROHIBICIONES', que contenía restricciones de uso del inmueble.

MODIFICACIONES:
- PRECIO Y PAGO: Alquiler inicial: $650.000 mensuales → Alquiler inicial: $750.000 mensuales.
- PRECIO Y PAGO: Medio: transferencia bancaria (CBU informado por la locadora) → Medio: efectivo al contado en el domicilio de la locadora.
- DEPÓSITO: Monto: $650.000 (1 mes de alquiler) → Monto: $750.000 (1 mes de alquiler)."
}
```

---

## Arquitectura del proyecto

```
agentes-auditores-contratos/
├── data/
│   └── test_contracts/
│       └── contrato_N/
│           ├── original/       # Imágenes del contrato original
│           └── modificado/     # Imágenes del contrato modificado o adenda
├── output/                     # JSONs generados por el sistema
├── src/
│   ├── agents/
│   │   ├── contextualization_agent.py   # Agente analista: analiza la estructura de ambos contratos y su correspondencia
│   │   └── extraction_agent.py          # Agente auditor: detecta y describe los cambios entre versiones
│   ├── exceptions.py           # Excepciones personalizadas
│   ├── image_parser.py         # Transcripción de imágenes a texto (base64 + visión)
│   ├── logger.py               # Logger personalizado
│   ├── main.py                 # Script principal con los nodos de LangGraph
│   ├── models.py               # Estado compartido entre nodos y modelo de output validado con Pydantic
│   ├── output_writer.py        # Guarda el output del agente auditor como archivo .json
│   └── validations.py          # Validaciones de entrada en main
├── tests/
│   ├── golden/                 # Golden cases: transcripciones de referencia para tests
│   ├── test_transcriptor.py    # Tests de los Golden cases
│   └── test_validations.py     # Pruebas unitarias de validaciones
├── .env                        # Variables de entorno (no incluido en el repo)
├── pyproject.toml
└── README.md
```

---

## Stack tecnológico

| Tecnología | Rol en el proyecto |
|---|---|
| Python 3.12 | Lenguaje principal |
| LangGraph | Orquestación del flujo de agentes como grafo de nodos |
| LangChain | Integración con modelos de lenguaje y cadenas de prompts |
| OpenAI | Modelo de lenguaje para transcripción y análisis |
| Pydantic | Validación y tipado del output JSON |
| Langfuse | Observabilidad y trazabilidad de las llamadas a LLM |
| uv | Gestión de entorno virtual y dependencias |
| pytest | Framework de testing |

---

## Configuración del entorno

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```env
OPENAI_API_KEY=tu-api-key-aqui

LANGFUSE_SECRET_KEY=tu-secret-key
LANGFUSE_PUBLIC_KEY=tu-public-key
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

- Obtener API Key de OpenAI: [platform.openai.com](https://platform.openai.com)
- Obtener credenciales de Langfuse: [cloud.langfuse.com](https://cloud.langfuse.com)

---

## Instalación

**Clonar el repositorio**

```bash
git clone https://github.com/iv-chucoff/agentes-auditores-contratos.git
cd agentes-auditores-contratos
```

**Crear el entorno virtual**

```bash
uv venv
```

**Activar el entorno**

```bash
.venv\Scripts\activate.ps1
```

**Instalar dependencias**

```bash
uv sync
```

---

## Ejecución

Respetar el formato de carpetas para los contratos:

```
data/test_contracts/contrato_N/
    ├── original/      ← imágenes del contrato original
    └── modificado/    ← imágenes del contrato modificado
```

Ejecutar con:

```bash
python src/main.py data/test_contracts/contrato_N
```

Ejemplo:

```bash
python src/main.py data/test_contracts/contrato_1
```

El output se guarda automáticamente en la carpeta `output/` con el nombre `contrato_N_<timestamp>.json`.

---

## Tests

**Correr todos los tests**

```bash
uv run pytest tests/ -v
```

**Correr los Golden cases**

```bash
uv run pytest tests/test_transcriptor.py -v
```

**Correr pruebas unitarias**

```bash
uv run pytest tests/test_validations.py -v
```

---

## Contacto

**Autor:** Ivana Chucoff

LinkedIn: https://www.linkedin.com/in/ivanachucoff
