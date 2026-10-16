# Skale

Copyright © 2026 Pablo Olivares Rodriguez.

Skale is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE) (GNU AGPL v3).

Skale es software libre: puedes redistribuirlo y modificarlo según la GNU Affero General Public License v3 o posterior.
Se ofrece sin ninguna garantía. Código fuente: https://github.com/pabloor/SKALE

## Third-party components / componentes de terceros

- **JUCE 8** (plugin): GNU AGPL v3 (o licencia comercial). Se descarga al configurar; no está en el repositorio.
- **Celemony ARA SDK 2.3.0** (plugin): Apache License 2.0. Se descarga al configurar.
- **dr_wav, dr_mp3** (David Reid, `third_party/dr_libs`): dominio público o MIT-0, a elección.

## Trained models / modelos entrenados

Los pesos de las redes (`models/`, `src/analysis/KeyCnnWeights.h`) se entrenaron con conjuntos de investigación
publicados, cuyo audio no se distribuye aquí: FMAK/FMAKv2 (etiquetas CC BY 4.0; audio de Free Music Archive con la
licencia Creative Commons de cada pista), Beatport EDM Key (CC BY-SA 4.0), GiantSteps Key+, GuitarSet (CC BY 4.0) y
otros conjuntos pequeños de licencia abierta (ver `tools/fetch_datasets.py`). GTZAN solo se usó para evaluar.
