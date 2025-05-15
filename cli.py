import typer
import asyncio
import yaml
from pathlib import Path
from pydantic import ValidationError

from multimodal_rag.pipeline.indexer import run_index_pipeline
from multimodal_rag.pipeline.rag import run_rag_pipeline, RAGRequest
from multimodal_rag.config.schema import IndexingConfig, RAGConfig

app = typer.Typer(help="Multimodal RAG CLI")


@app.command()
def index(
    source: str = typer.Argument(..., help="Input source: local folder, archive file, or GitHub repo URL"),
    config_path: Path = typer.Argument(..., help="Path to the JSON or YAML config file"),
    project_id: str = typer.Argument(..., help="Project ID to associate with the indexed documents")
):
    """Indexing pipeline: load, chunk, embed, index"""
    try:
        cfg_dict = yaml.safe_load(config_path.read_text())
        cfg = IndexingConfig.parse_obj(cfg_dict)
    except ValidationError as e:
        typer.echo("Config validation failed:", err=True)
        typer.echo(e, err=True)
        raise typer.Exit(code=1)

    asyncio.run(run_index_pipeline(source, cfg, project_id))


@app.command()
def rag(
    config_path: Path = typer.Argument(..., help="Path to the YAML infrastructure config"),
    payload_path: Path = typer.Argument(..., help="Path to the JSON payload file with request params"),
    project_id: str = typer.Argument(..., help="Project ID to query from"),
    stream: bool = typer.Option(False, "--stream", help="Enable streaming response")
):
    """RAG pipeline: retrieve + generate based on config and payload"""

    try:
        cfg = RAGConfig.parse_obj(yaml.safe_load(config_path.read_text()))
        request = RAGRequest.parse_raw(payload_path.read_text())
    except ValidationError as e:
        typer.echo("Validation failed:", err=True)
        typer.echo(e, err=True)
        raise typer.Exit(code=1)

    asyncio.run(run_rag_pipeline(
        config=cfg,
        project_id=project_id,
        request=request,
        stream=stream,
    ))


if __name__ == "__main__":
    app()
