"""Routes pour l'export et le reporting."""

import logging
from io import BytesIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from industrial_inspection.reporting.excel_builder import ExcelHistoryBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["Export"])


class HistoryExportRequest(BaseModel):
    """Requête pour l'export de l'historique."""

    history_data: list[dict]


@router.post("/history-excel")
async def export_history_to_excel(request: HistoryExportRequest):
    """Génère un fichier Excel avec l'historique complet des inspections."""
    try:
        logger.info(f"Exporting {len(request.history_data)} inspection records to Excel")

        # Créer le builder
        builder = ExcelHistoryBuilder()

        # Construire le workbook
        workbook = builder.build(request.history_data)

        # Sauvegarder dans un buffer
        buffer = BytesIO()
        workbook.save(buffer)
        buffer.seek(0)

        logger.info("Excel export completed successfully")

        # Retourner le fichier
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=inspection_history.xlsx"},
        )

    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
