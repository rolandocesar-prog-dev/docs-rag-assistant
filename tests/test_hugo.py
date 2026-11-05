import pytest

from rag_assistant.adapters.loaders.hugo import strip_hugo


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('Un {{< glossary_tooltip text="Pod" term_id="pod" >}} corre.', "Un Pod corre."),
        ('Un {{< glossary_tooltip term_id="pod" >}} corre.', "Un pod corre."),
        ('los {{< glossary_tooltip\ntext="CSI" term_id="csi" >}}.', "los CSI."),
        ('Ver {{< link text="servicios" url="es/docs/service/" >}}.', "Ver servicios."),
        ('{{< figure src="a.svg" title="Figura 1." caption="Un Pod" >}}', "Un Pod"),
        ('{{< figure src="a.svg" title="Diagrama" >}}', "Diagrama"),
        ('{{% tab name="Nube" %}}texto{{% /tab %}}', "Nubetexto"),
        (
            '{{< feature-state for_k8s_version="v1.20" state="stable" >}}',
            "Estado de la función: Kubernetes v1.20 [stable]",
        ),
        ('{{< feature-state feature_gate_name="X" >}}', ""),
        ('## {{% heading "whatsnext" %}}', "## ¿Qué sigue?"),
        ('## {{% heading "prerequisites" %}}', "## prerequisites"),
        ("{{< note >}}\nImportante.\n{{< /note >}}", "\nImportante.\n"),
        ("{{<note>}}Sin espacios{{</note>}}", "Sin espacios"),
        ("{{< note >}}Cierre con espacio{{</ note >}}", "Cierre con espacio"),
        ('kubectl v{{< param "version" >}}', "kubectl v"),
        ('{{% code_sample file="pod.yaml" %}}', ""),
        ("a{{< comment >}}\nborrador\n{{< /comment >}}b", "ab"),
        ("Texto sin etiquetas.", "Texto sin etiquetas."),
    ],
)
def test_strip_hugo(source: str, expected: str) -> None:
    assert strip_hugo(source) == expected
