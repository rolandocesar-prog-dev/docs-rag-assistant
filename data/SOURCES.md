# Fuentes del corpus

Documentos públicos que usa el asistente y su licencia (requisito §3 de
[`specs/requirements.md`](../specs/requirements.md)). Ningún documento procede de clientes ni del
proyecto de grado.

| Corpus                       | Carpeta                       | Licencia                                                   | Archivos |
| ---------------------------- | ----------------------------- | ---------------------------------------------------------- | -------- |
| Kubernetes, «Conceptos» (ES) | `corpus/kubernetes/concepts/` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  | 87 `.md` |

## Kubernetes, «Conceptos» (ES)

- **Obra:** documentación de Kubernetes en español, sección «Conceptos».
- **Autoría:** © The Kubernetes Authors.
- **Origen:** [`kubernetes/website`](https://github.com/kubernetes/website), carpeta
  [`content/es/docs/concepts`](https://github.com/kubernetes/website/tree/f99b6afdfd42f38937b8f17b3015551a33dede40/content/es/docs/concepts),
  publicada en <https://kubernetes.io/es/docs/concepts/>.
- **Versión:** commit `f99b6afdfd42f38937b8f17b3015551a33dede40` (2026-11-05).
- **Licencia:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/);
  texto completo en [`corpus/kubernetes/LICENSE`](./corpus/kubernetes/LICENSE).
- **Cambios:** ninguno. Los archivos son copia exacta del commit indicado; la limpieza de metadatos
  y etiquetas de Hugo la hace el cargador al leerlos.

### Cómo actualizar la copia

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/kubernetes/website.git k8s-website
git -C k8s-website sparse-checkout set content/es/docs/concepts
rm -rf data/corpus/kubernetes/concepts && mkdir -p data/corpus/kubernetes/concepts
git -C k8s-website -c core.autocrlf=false archive HEAD:content/es/docs/concepts \
  | tar -x -C data/corpus/kubernetes/concepts
git -C k8s-website cat-file blob HEAD:LICENSE > data/corpus/kubernetes/LICENSE
```

Después, registrar aquí el nuevo commit (`git -C k8s-website rev-parse HEAD`) y el número de
archivos, y volver a correr la evaluación (RF-06).
