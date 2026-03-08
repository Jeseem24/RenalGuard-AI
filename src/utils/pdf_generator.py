"""
RenalGuard AI - Professional PDF Report Generator
"""

from fpdf import FPDF
from datetime import datetime
import os
from typing import Dict, List, Any


class ClinicalReportGenerator:
    """Generates professional, verifiable clinical screening reports."""

    # Colors
    INDIGO = (79, 70, 229)
    DARK = (30, 41, 59)
    MUTED = (100, 116, 139)
    WHITE = (255, 255, 255)
    LIGHT_BG = (248, 250, 252)
    GREEN = (16, 185, 129)
    AMBER = (245, 158, 11)
    RED = (239, 68, 68)
    LIGHT_GREEN = (209, 250, 229)
    LIGHT_AMBER = (254, 243, 199)
    LIGHT_RED = (254, 226, 226)

    TEST_INFO = {
        'sc':   ('Serum Creatinine',    'mg/dL',        '0.6-1.2'),
        'bu':   ('Blood Urea',          'mg/dL',        '7-20'),
        'hemo': ('Hemoglobin',          'g/dL',         '12-17'),
        'bp':   ('Blood Pressure',      'mmHg',         '80-120'),
        'bgr':  ('Blood Glucose',       'mg/dL',        '70-140'),
        'sod':  ('Sodium',              'mEq/L',        '136-145'),
        'pot':  ('Potassium',           'mEq/L',        '3.5-5.0'),
        'pcv':  ('Packed Cell Volume',  '%',            '36-50'),
        'wc':   ('WBC Count',           'cells/cumm',   '4500-11000'),
        'rc':   ('RBC Count',           'M/cmm',        '4.5-5.5'),
        'sg':   ('Specific Gravity',    '',             '1.005-1.025'),
        'al':   ('Albumin (Urine)',     '',             '0'),
        'su':   ('Sugar (Urine)',       '',             '0'),
        'age':  ('Age',                 'years',        '-'),
    }

    STAGE_RECS = {
        1: [
            'Maintain healthy lifestyle with regular exercise (150 min/week)',
            'Monitor blood pressure regularly (target <130/80 mmHg)',
            'Control blood sugar if diabetic (HbA1c <7%)',
            'Annual kidney function testing (creatinine, eGFR, urinalysis)',
            'Stay hydrated and eat a balanced, low-sodium diet',
            'Avoid nephrotoxic agents (NSAIDs, herbal supplements)',
        ],
        2: [
            'Monitor kidney function every 6-12 months',
            'Strict blood pressure control (<130/80 mmHg) - consider ACE/ARB',
            'Dietary sodium restriction (<2g/day)',
            'Moderate protein intake (0.8g/kg/day)',
            'Regular cardiovascular risk assessment',
            'Avoid NSAIDs and contrast dye exposure',
        ],
        3: [
            'Nephrology consultation recommended within 3 months',
            'Monitor kidney function every 3-6 months',
            'Strict blood pressure and blood sugar control',
            'Dietary modifications: low sodium, controlled protein, limit K+/PO4',
            'Screen for anemia, bone mineral disorders, acidosis',
            'Medication review for kidney-safe alternatives',
            'Cardiovascular risk management',
        ],
        4: [
            'URGENT: Nephrology referral within 1 month',
            'Discuss renal replacement therapy options (dialysis modalities)',
            'Evaluate kidney transplant candidacy',
            'Monthly kidney function monitoring',
            'Strict dietary restrictions (protein, potassium, phosphorus, fluid)',
            'Manage complications: anemia (EPO), bone disease, acidosis',
            'Vascular access planning if dialysis anticipated',
        ],
        5: [
            'IMMEDIATE nephrology consultation required',
            'Initiate or continue dialysis as indicated',
            'Kidney transplant evaluation if not already done',
            'Urgent fluid and electrolyte management',
            'Dialysis-specific dietary protocol',
            'Comprehensive comorbidity management',
            'Psychosocial support and patient education',
        ],
    }

    def __init__(self):
        self.report_id = f"RG-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def generate_report(self, patient_data: Dict, prediction: Dict,
                        explanation: Dict, output_path: str = None) -> str:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # Page 1
        pdf.add_page()
        self._header(pdf)
        self._patient_section(pdf, patient_data)
        self._results_section(pdf, prediction)
        self._test_table(pdf, patient_data)

        # Page 2
        pdf.add_page()
        self._header_mini(pdf)
        self._explanation_section(pdf, explanation)
        self._recommendations_section(pdf, prediction)
        self._disclaimer_footer(pdf)

        if output_path is None:
            output_path = f"reports/CKD_Report_{self.report_id}.pdf"
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pdf.output(output_path)
        return output_path

    def _header(self, pdf: FPDF):
        # Brand
        pdf.set_font('Helvetica', 'B', 26)
        pdf.set_text_color(*self.INDIGO)
        pdf.cell(0, 14, 'RenalGuard AI', ln=True, align='C')
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(*self.MUTED)
        pdf.cell(0, 7, 'AI-Assisted Chronic Kidney Disease Screening Report', ln=True, align='C')
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 6, f'Report ID: {self.report_id}  |  {datetime.now().strftime("%B %d, %Y at %H:%M")}', ln=True, align='C')
        pdf.ln(4)
        # Disclaimer banner
        pdf.set_fill_color(254, 243, 199)
        pdf.set_text_color(133, 77, 14)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.multi_cell(0, 4.5,
            'CLINICAL DISCLAIMER: This is an AI-assisted screening tool - NOT a medical diagnosis. '
            'All results must be validated by a qualified nephrologist before clinical decisions are made.',
            align='C', fill=True)
        pdf.set_text_color(*self.DARK)
        pdf.ln(8)

    def _header_mini(self, pdf: FPDF):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*self.INDIGO)
        pdf.cell(0, 7, f'RenalGuard AI  |  Report {self.report_id}  |  Page {pdf.page_no()}', ln=True, align='C')
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

    def _section_title(self, pdf: FPDF, title: str):
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(*self.DARK)
        pdf.cell(0, 9, title, ln=True)
        pdf.set_draw_color(*self.INDIGO)
        pdf.set_line_width(0.6)
        pdf.line(10, pdf.get_y(), 80, pdf.get_y())
        pdf.set_line_width(0.2)
        pdf.ln(4)

    def _patient_section(self, pdf: FPDF, patient_data: Dict):
        self._section_title(pdf, 'Patient Information')

        # Get patient name
        patient_name = patient_data.get('patient_name', 'Not Provided')

        items = [
            ('Patient Name', patient_name),
            ('Age', f"{patient_data.get('age', 'N/A')} years"),
            ('Screening Type', 'CKD Risk Assessment - 24-parameter panel'),
            ('Date', datetime.now().strftime('%B %d, %Y')),
            ('Comorbidities',
             ', '.join(filter(None, [
                 'Hypertension' if patient_data.get('htn') == 'yes' else None,
                 'Diabetes' if patient_data.get('dm') == 'yes' else None,
                 'CAD' if patient_data.get('cad') == 'yes' else None,
                 'Anemia' if patient_data.get('ane') == 'yes' else None,
             ])) or 'None reported'),
        ]
        for label, val in items:
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(50, 7, f'{label}:')
            pdf.set_font('Helvetica', '', 10)
            # Highlight patient name
            if label == 'Patient Name':
                pdf.set_font('Helvetica', 'B', 11)
                pdf.set_text_color(*self.INDIGO)
                pdf.cell(0, 7, val, ln=True)
                pdf.set_text_color(*self.DARK)
            else:
                pdf.cell(0, 7, val, ln=True)
        pdf.ln(5)

    def _results_section(self, pdf: FPDF, prediction: Dict):
        self._section_title(pdf, 'Screening Results')
        risk = prediction.get('risk_level', 'UNKNOWN')
        color_map = {'LOW': self.LIGHT_GREEN, 'MODERATE': self.LIGHT_AMBER, 'HIGH': self.LIGHT_RED}
        text_map = {'LOW': (5, 122, 85), 'MODERATE': (133, 77, 14), 'HIGH': (153, 27, 27)}
        pdf.set_fill_color(*color_map.get(risk, self.LIGHT_AMBER))
        pdf.set_text_color(*text_map.get(risk, self.DARK))
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 11, f'  CKD RISK LEVEL:  {risk}', ln=True, fill=True)
        pdf.set_text_color(*self.DARK)
        pdf.ln(4)

        # Results grid
        grid = [
            ('Risk Score', f"{prediction.get('risk_score', 0):.1f} / 100"),
            ('CKD Stage', f"Stage {prediction.get('stage', '?')}"),
            ('eGFR', f"{prediction.get('egfr', '?')} mL/min"),
            ('Confidence', f"{prediction.get('confidence', 0):.1f}%"),
        ]
        col_w = 47.5
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_fill_color(*self.LIGHT_BG)
        for label, _ in grid:
            pdf.cell(col_w, 6, label, border=1, align='C', fill=True)
        pdf.ln()
        pdf.set_font('Helvetica', 'B', 11)
        for _, val in grid:
            pdf.cell(col_w, 9, val, border=1, align='C')
        pdf.ln(8)

    def _test_table(self, pdf: FPDF, patient_data: Dict):
        self._section_title(pdf, 'Laboratory Values')
        headers = ['Test Parameter', 'Value', 'Normal Range', 'Status']
        widths = [60, 35, 45, 50]

        pdf.set_fill_color(*self.INDIGO)
        pdf.set_text_color(*self.WHITE)
        pdf.set_font('Helvetica', 'B', 9)
        for h, w in zip(headers, widths):
            pdf.cell(w, 8, h, border=1, align='C', fill=True)
        pdf.ln()

        pdf.set_text_color(*self.DARK)
        for i, (key, (name, unit, normal)) in enumerate(self.TEST_INFO.items()):
            if key not in patient_data:
                continue
            val = patient_data[key]
            val_str = f"{val} {unit}".strip() if unit else str(val)
            status = self._calc_status(val, normal)

            pdf.set_fill_color(*self.LIGHT_BG if i % 2 == 0 else self.WHITE)
            pdf.set_font('Helvetica', '', 9)
            pdf.cell(widths[0], 7, name, border=1, fill=True)
            pdf.cell(widths[1], 7, val_str, border=1, align='C', fill=True)
            pdf.cell(widths[2], 7, normal, border=1, align='C', fill=True)

            if status in ('HIGH', 'LOW'):
                pdf.set_text_color(*self.RED)
            elif status == 'NORMAL':
                pdf.set_text_color(*self.GREEN)
            else:
                pdf.set_text_color(*self.MUTED)
            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(widths[3], 7, status, border=1, align='C', fill=True)
            pdf.set_text_color(*self.DARK)
            pdf.ln()
        pdf.ln(4)

    @staticmethod
    def _calc_status(value, normal_range: str) -> str:
        if normal_range in ('-', '', '0'):
            if normal_range == '0':
                try:
                    return 'NORMAL' if float(value) == 0 else 'ELEVATED'
                except (ValueError, TypeError):
                    return 'N/A'
            return 'N/A'
        for sep in ['–', '-']:
            if sep in normal_range:
                parts = normal_range.split(sep)
                if len(parts) == 2:
                    try:
                        lo, hi = float(parts[0]), float(parts[1])
                        v = float(value)
                        if v < lo:
                            return 'LOW'
                        elif v > hi:
                            return 'HIGH'
                        return 'NORMAL'
                    except (ValueError, TypeError):
                        return 'N/A'
        return 'N/A'

    def _explanation_section(self, pdf: FPDF, explanation: Dict):
        self._section_title(pdf, 'AI Risk Factor Analysis')
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5.5,
            'The AI model identified the following biomarkers as the primary drivers '
            'of this risk assessment, ranked by contribution magnitude:')
        pdf.ln(3)

        factors = explanation.get('top_risk_factors', [])[:6]
        if factors:
            for i, f in enumerate(factors, 1):
                feat = f.get('feature', '?').upper()
                val = f.get('feature_value', '?')
                pct = f.get('contribution_pct', 0)
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 7, f"{i}. {feat}", ln=True)
                pdf.set_font('Helvetica', '', 9)
                pdf.cell(0, 5, f"    Patient Value: {val}  |  Risk Contribution: {pct:.1f}%", ln=True)
                pdf.ln(2)
        else:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.cell(0, 7, 'Detailed factor analysis not available for this screening.', ln=True)
        pdf.ln(4)

    def _recommendations_section(self, pdf: FPDF, prediction: Dict):
        self._section_title(pdf, 'Clinical Recommendations')
        stage = prediction.get('stage', 3)
        recs = self.STAGE_RECS.get(stage, self.STAGE_RECS[3])

        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, f'Based on Stage {stage} Assessment (KDIGO Guidelines):', ln=True)
        pdf.ln(2)

        pdf.set_font('Helvetica', '', 10)
        for i, rec in enumerate(recs, 1):
            pdf.multi_cell(0, 5.5, f"  {i}. {rec}")
            pdf.ln(1)
        pdf.ln(4)

    def _disclaimer_footer(self, pdf: FPDF):
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.set_text_color(*self.MUTED)
        pdf.multi_cell(0, 3.8,
            'This report was generated by RenalGuard AI, an AI-assisted clinical decision support tool. '
            'It is NOT a medical diagnosis. This report should be reviewed by a qualified nephrologist '
            'or physician before any clinical decisions are made. The AI model was trained on the '
            'UCI Chronic Kidney Disease dataset and uses the CKD-EPI 2021 equation for eGFR estimation. '
            'Results are probabilistic and must be confirmed with clinical evaluation.')
        pdf.ln(3)
        pdf.set_font('Helvetica', '', 7)
        pdf.cell(0, 4,
            f'Report ID: {self.report_id}  |  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  |  RenalGuard AI v2.0',
            align='C')