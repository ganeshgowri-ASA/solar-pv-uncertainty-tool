"""
Professional Report Generator for PV Uncertainty Analysis
Generates ISO 17025 compliant reports in PDF and Excel formats
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import xlsxwriter


class ISO17025ReportGenerator:
    """
    Generate ISO 17025 compliant calibration/test reports.
    """

    def __init__(
        self,
        company_name: str = "PV Testing Laboratory",
        document_format: str = "PV-UNC-001",
        lab_address: str = "",
        lab_accreditation: str = "",
        lab_logo_path: Optional[str] = None
    ):
        """
        Initialize report generator.

        Args:
            company_name: Laboratory name
            document_format: Document format number
            lab_address: Laboratory address
            lab_accreditation: Accreditation details
            lab_logo_path: Path to laboratory logo image
        """
        self.company_name = company_name
        self.document_format = document_format
        self.lab_address = lab_address
        self.lab_accreditation = lab_accreditation
        self.lab_logo_path = lab_logo_path

        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=10,
            fontName='Helvetica-Bold',
            leftIndent=0
        ))

        # Report text
        self.styles.add(ParagraphStyle(
            name='ReportText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        ))

    def generate_pdf_report(
        self,
        uncertainty_result: Dict,
        module_config: Dict,
        simulator_config: Dict,
        reference_config: Dict,
        measurement_data: Dict,
        report_config: Dict,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Generate comprehensive PDF report.

        Args:
            uncertainty_result: Calculated uncertainty results
            module_config: Module configuration
            simulator_config: Sun simulator configuration
            reference_config: Reference device configuration
            measurement_data: Measurement data
            report_config: Report metadata (preparer, reviewer, etc.)
            output_path: Optional file path to save PDF

        Returns:
            PDF file as bytes
        """
        # Create PDF buffer
        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Story (content)
        story = []

        # Header
        story.extend(self._create_header(report_config))

        # Title
        title = Paragraph(
            "PV MODULE MEASUREMENT UNCERTAINTY REPORT",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 12))

        # Report information
        story.extend(self._create_report_info(report_config))
        story.append(Spacer(1, 12))

        # Module information
        story.append(Paragraph("1. MODULE INFORMATION", self.styles['SectionHeader']))
        story.extend(self._create_module_info(module_config))
        story.append(Spacer(1, 12))

        # Test equipment
        story.append(Paragraph("2. TEST EQUIPMENT", self.styles['SectionHeader']))
        story.extend(self._create_equipment_info(simulator_config, reference_config))
        story.append(Spacer(1, 12))

        # Measurement results
        story.append(Paragraph("3. MEASUREMENT RESULTS", self.styles['SectionHeader']))
        story.extend(self._create_measurement_results(measurement_data))
        story.append(Spacer(1, 12))

        # Uncertainty analysis
        story.append(Paragraph("4. UNCERTAINTY ANALYSIS", self.styles['SectionHeader']))
        story.extend(self._create_uncertainty_section(uncertainty_result))

        # Page break
        story.append(PageBreak())

        # Uncertainty budget
        story.append(Paragraph("5. DETAILED UNCERTAINTY BUDGET", self.styles['SectionHeader']))
        story.extend(self._create_uncertainty_budget_table(uncertainty_result))
        story.append(Spacer(1, 12))

        # Statement of compliance
        story.append(Paragraph("6. STATEMENT OF CONFORMITY", self.styles['SectionHeader']))
        story.extend(self._create_conformity_statement(uncertainty_result))
        story.append(Spacer(1, 12))

        # Signatures
        story.extend(self._create_signature_section(report_config))

        # Build PDF
        doc.build(story)

        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()

        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        return pdf_bytes

    def _create_header(self, report_config: Dict) -> List:
        """Create report header with logo and document control."""
        elements = []

        # Create header table
        header_data = [
            [
                self.company_name,
                "Document Format:",
                self.document_format
            ],
            [
                self.lab_address if self.lab_address else "",
                "Record Ref:",
                report_config.get('record_ref', 'PV-TEST-001')
            ],
            [
                self.lab_accreditation if self.lab_accreditation else "",
                "Date:",
                report_config.get('test_date', datetime.now().strftime('%Y-%m-%d'))
            ]
        ]

        header_table = Table(header_data, colWidths=[3.5*inch, 1.2*inch, 1.5*inch])
        header_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor('#1e3a8a')),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.grey)
        ]))

        elements.append(header_table)
        elements.append(Spacer(1, 12))

        return elements

    def _create_report_info(self, report_config: Dict) -> List:
        """Create report information section."""
        elements = []

        info_data = [
            ["Report Number:", report_config.get('report_number', 'AUTO-GEN-001')],
            ["Client:", report_config.get('client_name', 'N/A')],
            ["Test Date:", report_config.get('test_date', datetime.now().strftime('%Y-%m-%d'))],
            ["Report Date:", datetime.now().strftime('%Y-%m-%d %H:%M')]
        ]

        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))

        elements.append(info_table)

        return elements

    def _create_module_info(self, module_config: Dict) -> List:
        """Create module information section."""
        elements = []

        module_data = [
            ["Technology:", module_config.get('technology', 'N/A')],
            ["Nameplate Power:", f"{module_config.get('pmax_nameplate', 0):.2f} W"],
            ["Module Area:", f"{module_config.get('module_area', 0):.2f} m²"],
            ["Temperature Coefficient (γ_Pmax):", f"{module_config.get('gamma_pmax', 0):.3f} %/°C"],
        ]

        if module_config.get('cells_series', 0) > 0:
            module_data.append([
                "Cell Configuration:",
                f"{module_config.get('cells_series', 0)}S × {module_config.get('cells_parallel', 1)}P"
            ])

        module_table = Table(module_data, colWidths=[2.5*inch, 3.5*inch])
        module_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8fafc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))

        elements.append(module_table)

        return elements

    def _create_equipment_info(self, simulator_config: Dict, reference_config: Dict) -> List:
        """Create equipment information section."""
        elements = []

        # Sun simulator
        sim_data = [
            ["Sun Simulator", ""],
            ["Manufacturer:", simulator_config.get('manufacturer', 'N/A')],
            ["Model:", simulator_config.get('model', 'N/A')],
            ["Classification:", simulator_config.get('classification', 'N/A')],
            ["Spatial Non-uniformity:", f"{simulator_config.get('uniformity', 0):.1f}%"],
            ["Temporal Instability:", f"{simulator_config.get('temporal_instability', 0):.1f}%"]
        ]

        # Reference device
        ref_data = [
            ["Reference Device", ""],
            ["Type:", reference_config.get('ref_type', 'N/A')],
            ["Calibration Lab:", reference_config.get('lab_name', 'N/A')],
            ["Calibration Uncertainty:", f"{reference_config.get('calibration_uncertainty', 0):.2f}%"]
        ]

        combined_data = sim_data + [[""]*2] + ref_data

        equipment_table = Table(combined_data, colWidths=[2.5*inch, 3.5*inch])
        equipment_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 6), (0, 6), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 1), (0, 5), 'Helvetica-Bold'),
            ('FONTNAME', (0, 7), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
            ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor('#dbeafe')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('SPAN', (0, 0), (-1, 0)),
            ('SPAN', (0, 6), (-1, 6))
        ]))

        elements.append(equipment_table)

        return elements

    def _create_measurement_results(self, measurement_data: Dict) -> List:
        """Create measurement results section."""
        elements = []

        # STC conditions statement
        conditions_text = Paragraph(
            "<b>Test Conditions:</b> Standard Test Conditions (STC) - 1000 W/m², 25°C, AM1.5G spectrum",
            self.styles['ReportText']
        )
        elements.append(conditions_text)
        elements.append(Spacer(1, 6))

        # Results table
        results_data = [
            ["Parameter", "Value", "Unit"],
            ["Open Circuit Voltage (Voc)", f"{measurement_data.get('voc', 0):.2f}", "V"],
            ["Short Circuit Current (Isc)", f"{measurement_data.get('isc', 0):.2f}", "A"],
            ["Voltage at MPP (Vmp)", f"{measurement_data.get('vmp', 0):.2f}", "V"],
            ["Current at MPP (Imp)", f"{measurement_data.get('imp', 0):.2f}", "A"],
            ["Maximum Power (Pmax)", f"{measurement_data.get('pmax', 0):.2f}", "W"],
            ["Fill Factor (FF)", f"{measurement_data.get('fill_factor', 0):.4f}", "-"]
        ]

        results_table = Table(results_data, colWidths=[3*inch, 2*inch, 1*inch])
        results_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))

        elements.append(results_table)

        return elements

    def _create_uncertainty_section(self, uncertainty_result: Dict) -> List:
        """Create uncertainty analysis section."""
        elements = []

        pmax = uncertainty_result.get('pmax_measured', 0)
        combined_unc = uncertainty_result.get('combined_standard_uncertainty', 0)
        expanded_unc = uncertainty_result.get('expanded_uncertainty_k2', 0)
        unc_absolute = uncertainty_result.get('pmax_uncertainty_absolute', 0)
        ci_lower, ci_upper = uncertainty_result.get('pmax_confidence_interval_95', (0, 0))

        # Summary table
        summary_data = [
            ["Combined Standard Uncertainty (k=1)", f"{combined_unc:.2f}%"],
            ["Expanded Uncertainty (k=2)", f"{expanded_unc:.2f}%"],
            ["Absolute Uncertainty", f"±{unc_absolute:.2f} W"],
            ["95% Confidence Interval", f"[{ci_lower:.2f} W, {ci_upper:.2f} W]"]
        ]

        summary_table = Table(summary_data, colWidths=[3.5*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#dbeafe')),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))

        elements.append(summary_table)
        elements.append(Spacer(1, 12))

        # Measurement result statement
        statement_text = f"""
        <b>Measurement Result:</b><br/>
        The maximum power of the tested photovoltaic module at Standard Test Conditions
        (1000 W/m², 25°C, AM1.5G spectrum) is:<br/><br/>
        <font size="12" color="#dc2626"><b>Pmax = {pmax:.2f} W ± {unc_absolute:.2f} W (k=2)</b></font><br/><br/>
        The reported expanded uncertainty is based on a combined standard uncertainty
        multiplied by a coverage factor k=2, providing a level of confidence of
        approximately 95%. This uncertainty budget was calculated in accordance with
        JCGM 100:2008 (GUM).
        """

        statement = Paragraph(statement_text, self.styles['ReportText'])
        elements.append(statement)

        return elements

    def _create_uncertainty_budget_table(self, uncertainty_result: Dict) -> List:
        """Create detailed uncertainty budget table."""
        elements = []

        components = uncertainty_result.get('components', [])

        if not components:
            return elements

        # Table header
        table_data = [
            ["ID", "Uncertainty Source", "Std Unc", "Distribution", "Contrib %"]
        ]

        # Add top contributors
        for comp in components[:15]:  # Top 15
            table_data.append([
                comp.get('factor_id', ''),
                comp.get('name', ''),
                f"{comp.get('standard_uncertainty', 0):.4f}",
                comp.get('distribution', ''),
                f"{comp.get('percentage_contribution', 0):.2f}"
            ])

        # Create table
        budget_table = Table(
            table_data,
            colWidths=[0.6*inch, 3*inch, 1*inch, 1.2*inch, 1*inch]
        )

        budget_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))

        elements.append(budget_table)

        return elements

    def _create_conformity_statement(self, uncertainty_result: Dict) -> List:
        """Create statement of conformity section."""
        elements = []

        text = """
        This report provides the results of the measurement and the associated
        measurement uncertainty. No statement of conformity to a specification or
        standard is provided unless explicitly requested by the client and documented
        in the test plan.
        <br/><br/>
        The uncertainty analysis was performed in accordance with:
        <br/>
        • JCGM 100:2008 - Guide to the Expression of Uncertainty in Measurement (GUM)
        <br/>
        • JCGM 101:2008 - Supplement 1 to the GUM - Propagation of distributions using Monte Carlo method
        <br/>
        • IEC 60904-1:2020 - Photovoltaic devices - Part 1: Measurement of photovoltaic current-voltage characteristics
        """

        statement = Paragraph(text, self.styles['ReportText'])
        elements.append(statement)

        return elements

    def _create_signature_section(self, report_config: Dict) -> List:
        """Create signature section."""
        elements = []

        elements.append(Spacer(1, 24))

        # Signature table
        sig_data = [
            ["Prepared by:", "Reviewed by:", "Approved by:"],
            [
                report_config.get('preparer_name', '_______________'),
                report_config.get('reviewer_name', '_______________'),
                report_config.get('approver_name', '_______________')
            ],
            [
                report_config.get('preparer_date', '_______________'),
                report_config.get('reviewer_date', '_______________'),
                report_config.get('approver_date', '_______________')
            ]
        ]

        sig_table = Table(sig_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        sig_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
            ('LINEABOVE', (0, 1), (-1, 1), 0.5, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 1), 4)
        ]))

        elements.append(sig_table)

        # End of report statement
        elements.append(Spacer(1, 12))
        end_statement = Paragraph(
            "<i>*** END OF REPORT ***</i>",
            self.styles['ReportText']
        )
        end_statement.hAlign = 'CENTER'
        elements.append(end_statement)

        return elements

    def generate_excel_report(
        self,
        uncertainty_result: Dict,
        module_config: Dict,
        simulator_config: Dict,
        reference_config: Dict,
        measurement_data: Dict,
        report_config: Dict,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Generate Excel report with multiple sheets.

        Args:
            uncertainty_result: Calculated uncertainty results
            module_config: Module configuration
            simulator_config: Sun simulator configuration
            reference_config: Reference device configuration
            measurement_data: Measurement data
            report_config: Report metadata
            output_path: Optional file path to save Excel

        Returns:
            Excel file as bytes
        """
        # Create Excel buffer
        buffer = io.BytesIO()

        # Create workbook
        workbook = xlsxwriter.Workbook(buffer, {'in_memory': True})

        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#3b82f6',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })

        subheader_format = workbook.add_format({
            'bold': True,
            'bg_color': '#dbeafe',
            'border': 1
        })

        data_format = workbook.add_format({
            'border': 1,
            'align': 'left',
            'valign': 'vcenter'
        })

        number_format = workbook.add_format({
            'border': 1,
            'align': 'center',
            'num_format': '0.00'
        })

        # Sheet 1: Summary
        ws_summary = workbook.add_worksheet('Summary')
        self._create_excel_summary_sheet(
            ws_summary, uncertainty_result, module_config,
            measurement_data, header_format, data_format, number_format
        )

        # Sheet 2: Uncertainty Budget
        ws_budget = workbook.add_worksheet('Uncertainty Budget')
        self._create_excel_budget_sheet(
            ws_budget, uncertainty_result, header_format, data_format, number_format
        )

        # Sheet 3: Equipment
        ws_equipment = workbook.add_worksheet('Equipment')
        self._create_excel_equipment_sheet(
            ws_equipment, simulator_config, reference_config,
            header_format, data_format
        )

        # Close workbook
        workbook.close()

        # Get Excel bytes
        excel_bytes = buffer.getvalue()
        buffer.close()

        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(excel_bytes)

        return excel_bytes

    def _create_excel_summary_sheet(
        self, worksheet, uncertainty_result, module_config,
        measurement_data, header_fmt, data_fmt, num_fmt
    ):
        """Create summary sheet in Excel."""
        # Title
        worksheet.write('A1', 'PV MODULE MEASUREMENT UNCERTAINTY REPORT', header_fmt)
        worksheet.merge_range('A1:D1', 'PV MODULE MEASUREMENT UNCERTAINTY REPORT', header_fmt)

        row = 3

        # Module info
        worksheet.write(row, 0, 'MODULE INFORMATION', header_fmt)
        worksheet.merge_range(row, 0, row, 3, 'MODULE INFORMATION', header_fmt)
        row += 1

        module_info = [
            ['Technology:', module_config.get('technology', 'N/A')],
            ['Nameplate Power:', f"{module_config.get('pmax_nameplate', 0)} W"],
            ['Module Area:', f"{module_config.get('module_area', 0)} m²"]
        ]

        for label, value in module_info:
            worksheet.write(row, 0, label, data_fmt)
            worksheet.write(row, 1, value, data_fmt)
            row += 1

        row += 1

        # Measurement results
        worksheet.write(row, 0, 'MEASUREMENT RESULTS (STC)', header_fmt)
        worksheet.merge_range(row, 0, row, 3, 'MEASUREMENT RESULTS (STC)', header_fmt)
        row += 1

        results = [
            ['Voc (V):', measurement_data.get('voc', 0)],
            ['Isc (A):', measurement_data.get('isc', 0)],
            ['Vmp (V):', measurement_data.get('vmp', 0)],
            ['Imp (A):', measurement_data.get('imp', 0)],
            ['Pmax (W):', measurement_data.get('pmax', 0)],
            ['Fill Factor:', measurement_data.get('fill_factor', 0)]
        ]

        for label, value in results:
            worksheet.write(row, 0, label, data_fmt)
            worksheet.write(row, 1, value, num_fmt)
            row += 1

        row += 1

        # Uncertainty summary
        worksheet.write(row, 0, 'UNCERTAINTY SUMMARY', header_fmt)
        worksheet.merge_range(row, 0, row, 3, 'UNCERTAINTY SUMMARY', header_fmt)
        row += 1

        unc_summary = [
            ['Combined Standard Uncertainty (%):', uncertainty_result.get('combined_standard_uncertainty', 0)],
            ['Expanded Uncertainty k=2 (%):', uncertainty_result.get('expanded_uncertainty_k2', 0)],
            ['Absolute Uncertainty (W):', uncertainty_result.get('pmax_uncertainty_absolute', 0)],
        ]

        for label, value in unc_summary:
            worksheet.write(row, 0, label, data_fmt)
            worksheet.write(row, 1, value, num_fmt)
            row += 1

        # Set column widths
        worksheet.set_column('A:A', 35)
        worksheet.set_column('B:B', 20)

    def _create_excel_budget_sheet(
        self, worksheet, uncertainty_result, header_fmt, data_fmt, num_fmt
    ):
        """Create uncertainty budget sheet in Excel."""
        # Header
        headers = ['ID', 'Uncertainty Source', 'Std Uncertainty', 'Distribution',
                   'Sensitivity', 'Contribution (%)']

        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_fmt)

        # Data
        components = uncertainty_result.get('components', [])

        row = 1
        for comp in components:
            worksheet.write(row, 0, comp.get('factor_id', ''), data_fmt)
            worksheet.write(row, 1, comp.get('name', ''), data_fmt)
            worksheet.write(row, 2, comp.get('standard_uncertainty', 0), num_fmt)
            worksheet.write(row, 3, comp.get('distribution', ''), data_fmt)
            worksheet.write(row, 4, comp.get('sensitivity_coefficient', 0), num_fmt)
            worksheet.write(row, 5, comp.get('percentage_contribution', 0), num_fmt)
            row += 1

        # Set column widths
        worksheet.set_column('A:A', 8)
        worksheet.set_column('B:B', 40)
        worksheet.set_column('C:F', 15)

    def _create_excel_equipment_sheet(
        self, worksheet, simulator_config, reference_config, header_fmt, data_fmt
    ):
        """Create equipment sheet in Excel."""
        worksheet.write('A1', 'TEST EQUIPMENT INFORMATION', header_fmt)
        worksheet.merge_range('A1:B1', 'TEST EQUIPMENT INFORMATION', header_fmt)

        row = 3

        # Sun Simulator
        worksheet.write(row, 0, 'SUN SIMULATOR', header_fmt)
        worksheet.merge_range(row, 0, row, 1, 'SUN SIMULATOR', header_fmt)
        row += 1

        sim_info = [
            ['Manufacturer:', simulator_config.get('manufacturer', 'N/A')],
            ['Model:', simulator_config.get('model', 'N/A')],
            ['Classification:', simulator_config.get('classification', 'N/A')],
            ['Spatial Non-uniformity (%):', simulator_config.get('uniformity', 0)],
            ['Temporal Instability (%):', simulator_config.get('temporal_instability', 0)]
        ]

        for label, value in sim_info:
            worksheet.write(row, 0, label, data_fmt)
            worksheet.write(row, 1, value, data_fmt)
            row += 1

        row += 1

        # Reference Device
        worksheet.write(row, 0, 'REFERENCE DEVICE', header_fmt)
        worksheet.merge_range(row, 0, row, 1, 'REFERENCE DEVICE', header_fmt)
        row += 1

        ref_info = [
            ['Type:', reference_config.get('ref_type', 'N/A')],
            ['Calibration Lab:', reference_config.get('lab_name', 'N/A')],
            ['Calibration Uncertainty (%):', reference_config.get('calibration_uncertainty', 0)]
        ]

        for label, value in ref_info:
            worksheet.write(row, 0, label, data_fmt)
            worksheet.write(row, 1, value, data_fmt)
            row += 1

        # Set column widths
        worksheet.set_column('A:A', 35)
        worksheet.set_column('B:B', 30)
