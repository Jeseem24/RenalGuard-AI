"""
RenalGuard AI - PDF Report Generator
Generates clinical screening reports in PDF format
"""

from fpdf import FPDF
from datetime import datetime
import os
from typing import Dict, List, Any, Optional


class ClinicalReportGenerator:
    """
    Generates professional PDF clinical reports for CKD screening
    """
    
    def __init__(self):
        self.report_id = self._generate_report_id()
    
    def _generate_report_id(self) -> str:
        """Generate a unique report ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"RG-{timestamp}"
    
    def generate_report(
        self,
        patient_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
        explanation: Dict[str, Any],
        output_path: str = None
    ) -> str:
        """
        Generate a complete clinical screening report
        
        Args:
            patient_data: Dictionary of patient test values
            prediction_result: Dictionary with prediction results
            explanation: Dictionary with SHAP explanation
            output_path: Path to save the PDF (optional)
            
        Returns:
            Path to the generated PDF file
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Page 1 - Main Report
        pdf.add_page()
        self._add_header(pdf)
        self._add_patient_info(pdf, patient_data)
        self._add_results_summary(pdf, prediction_result)
        self._add_test_values_table(pdf, patient_data)
        
        # Page 2 - Explanation & Recommendations
        pdf.add_page()
        self._add_explanation_section(pdf, explanation)
        self._add_recommendations(pdf, prediction_result)
        self._add_footer(pdf)
        
        # Save the PDF
        if output_path is None:
            output_path = f"reports/CKD_Screening_Report_{self.report_id}.pdf"
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        pdf.output(output_path)
        return output_path
    
    def _add_header(self, pdf: FPDF):
        """Add report header"""
        # Title
        pdf.set_font('Helvetica', 'B', 24)
        pdf.set_text_color(26, 115, 232)  # Blue
        pdf.cell(0, 15, 'RenalGuard AI', ln=True, align='C')
        
        pdf.set_font('Helvetica', '', 14)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, 'AI-Assisted Kidney Health Screening Report', ln=True, align='C')
        
        # Report ID and Date
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, f'Report ID: {self.report_id}', ln=True, align='C')
        pdf.cell(0, 6, f'Date: {datetime.now().strftime("%B %d, %Y at %H:%M")}', ln=True, align='C')
        
        # Disclaimer
        pdf.set_fill_color(255, 243, 205)  # Light yellow
        pdf.set_text_color(133, 100, 4)  # Dark yellow
        pdf.set_font('Helvetica', 'I', 8)
        pdf.ln(5)
        pdf.multi_cell(0, 5, 
            'DISCLAIMER: This is an AI-assisted screening report. It is NOT a medical diagnosis. '
            'Please consult a qualified nephrologist for clinical evaluation.',
            align='C', fill=True)
        
        pdf.ln(10)
    
    def _add_patient_info(self, pdf: FPDF, patient_data: Dict):
        """Add patient information section"""
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, 'Patient Information', ln=True)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        # Patient details
        info_items = [
            ('Age', f"{patient_data.get('age', 'N/A')} years"),
            ('Screening Type', 'CKD Risk Assessment'),
            ('Screening Date', datetime.now().strftime('%B %d, %Y'))
        ]
        
        for label, value in info_items:
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(50, 7, f'{label}:', align='L')
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 7, value, ln=True)
        
        pdf.ln(5)
    
    def _add_results_summary(self, pdf: FPDF, prediction_result: Dict):
        """Add the main results summary"""
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Screening Results', ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Risk Level Box
        risk_level = prediction_result.get('risk_level', 'UNKNOWN')
        risk_score = prediction_result.get('risk_score', 0)
        stage = prediction_result.get('stage', 'N/A')
        egfr = prediction_result.get('egfr', 'N/A')
        
        # Color based on risk
        if risk_level == 'LOW':
            bg_color = (198, 239, 206)  # Light green
            text_color = (0, 100, 0)
        elif risk_level == 'MODERATE':
            bg_color = (255, 243, 205)  # Light yellow
            text_color = (156, 102, 0)
        elif risk_level == 'HIGH':
            bg_color = (255, 218, 193)  # Light orange
            text_color = (200, 80, 0)
        else:  # VERY HIGH
            bg_color = (255, 200, 200)  # Light red
            text_color = (180, 0, 0)
        
        pdf.set_fill_color(*bg_color)
        pdf.set_text_color(*text_color)
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 12, f'  CKD RISK LEVEL: {risk_level}', ln=True, fill=True)
        
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
        
        # Results grid
        pdf.set_font('Helvetica', '', 11)
        col_width = 47.5
        
        results = [
            ('Risk Score', f"{risk_score}/100"),
            ('CKD Stage', f"Stage {stage}"),
            ('eGFR', f"{egfr} mL/min/1.73m²"),
            ('Confidence', f"{prediction_result.get('confidence', 'N/A')}%")
        ]
        
        for label, value in results:
            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(col_width, 6, label + ':', border=1, align='C')
        pdf.ln()
        
        for label, value in results:
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(col_width, 8, value, border=1, align='C')
        pdf.ln(10)
    
    def _add_test_values_table(self, pdf: FPDF, patient_data: Dict):
        """Add table of test values"""
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Test Values', ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        # Table header
        pdf.set_fill_color(26, 115, 232)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 9)
        
        headers = ['Test Parameter', 'Patient Value', 'Normal Range', 'Status']
        col_widths = [55, 35, 50, 50]
        
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 8, header, border=1, align='C', fill=True)
        pdf.ln()
        
        # Test values with normal ranges
        test_info = {
            'sc': ('Serum Creatinine', 'mg/dL', '0.6-1.2'),
            'bu': ('Blood Urea', 'mg/dL', '7-20'),
            'hemo': ('Hemoglobin', 'gms', '12-17'),
            'bp': ('Blood Pressure', 'mm/Hg', '80-120'),
            'bgr': ('Blood Glucose', 'mg/dL', '70-140'),
            'sod': ('Sodium', 'mEq/L', '136-145'),
            'pot': ('Potassium', 'mEq/L', '3.5-5.0'),
            'pcv': ('Packed Cell Volume', '%', '36-50'),
            'wc': ('WBC Count', 'cells/cumm', '4500-11000'),
            'rc': ('RBC Count', 'millions/cmm', '4.5-5.5'),
            'sg': ('Specific Gravity', '', '1.005-1.025'),
            'al': ('Albumin', '', '0'),
            'age': ('Age', 'years', '-')
        }
        
        pdf.set_text_color(0, 0, 0)
        
        for key, (name, unit, normal) in test_info.items():
            if key in patient_data:
                value = patient_data[key]
                if unit:
                    value_str = f"{value} {unit}"
                else:
                    value_str = str(value)
                
                # Determine status
                status = self._get_status(key, value, normal)
                
                # Alternate row colors
                if list(test_info.keys()).index(key) % 2 == 0:
                    pdf.set_fill_color(248, 249, 250)
                else:
                    pdf.set_fill_color(255, 255, 255)
                
                pdf.set_font('Helvetica', '', 9)
                pdf.cell(col_widths[0], 7, name, border=1, fill=True)
                pdf.cell(col_widths[1], 7, str(value_str), border=1, align='C', fill=True)
                pdf.cell(col_widths[2], 7, normal, border=1, align='C', fill=True)
                
                # Status with color
                if 'HIGH' in status or 'LOW' in status:
                    pdf.set_text_color(200, 0, 0)
                else:
                    pdf.set_text_color(0, 150, 0)
                pdf.cell(col_widths[3], 7, status, border=1, align='C', fill=True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln()
        
        pdf.ln(5)
    
    def _get_status(self, key: str, value: float, normal_range: str) -> str:
        """Determine status of a test value"""
        if normal_range == '-' or not normal_range:
            return 'N/A'
        
        try:
            parts = normal_range.split('-')
            if len(parts) == 2:
                low = float(parts[0])
                high = float(parts[1])
                
                if value < low:
                    return 'LOW'
                elif value > high:
                    return 'HIGH'
                else:
                    return 'NORMAL'
        except:
            pass
        
        return 'N/A'
    
    def _add_explanation_section(self, pdf: FPDF, explanation: Dict):
        """Add AI explanation section"""
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'AI Explanation', ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, 
            'The AI model identified the following factors as most contributing to the prediction:')
        pdf.ln(3)
        
        if explanation and 'top_risk_factors' in explanation:
            for i, factor in enumerate(explanation['top_risk_factors'][:5], 1):
                feature = factor.get('feature', 'Unknown')
                value = factor.get('feature_value', 'N/A')
                contribution = factor.get('contribution_pct', 0)
                
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 7, f"{i}. {feature.upper()}", ln=True)
                pdf.set_font('Helvetica', '', 9)
                pdf.cell(0, 5, f"   Value: {value}", ln=True)
                pdf.cell(0, 5, f"   Contribution: {contribution:.1f}%", ln=True)
                pdf.ln(2)
        else:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.cell(0, 7, 'No detailed explanation available.', ln=True)
        
        pdf.ln(5)
    
    def _add_recommendations(self, pdf: FPDF, prediction_result: Dict):
        """Add recommendations based on results"""
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Recommendations', ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        stage = prediction_result.get('stage', 3)
        
        recommendations = self._get_stage_recommendations(stage)
        
        pdf.set_font('Helvetica', '', 10)
        for i, rec in enumerate(recommendations, 1):
            # Use numbered list instead of bullets for better compatibility
            pdf.multi_cell(0, 6, f"{i}. {rec}")
            pdf.ln(1)
        
        pdf.ln(5)
    
    def _get_stage_recommendations(self, stage: int) -> List[str]:
        """Get recommendations based on CKD stage"""
        recommendations = {
            1: [
                'Maintain healthy lifestyle with regular exercise',
                'Monitor blood pressure regularly',
                'Control blood sugar if diabetic',
                'Follow up with annual kidney function tests',
                'Stay hydrated and eat a balanced diet'
            ],
            2: [
                'Continue healthy lifestyle modifications',
                'Regular monitoring of kidney function (every 6-12 months)',
                'Control blood pressure (target <130/80 mmHg)',
                'Consider ACE inhibitors or ARBs if hypertensive',
                'Limit NSAID use'
            ],
            3: [
                'Nephrology consultation recommended within 3 months',
                'Regular monitoring every 3-6 months',
                'Strict blood pressure and blood sugar control',
                'Dietary modifications as advised by dietitian',
                'Review medications for kidney-friendly alternatives'
            ],
            4: [
                'URGENT: Nephrology referral within 1 month',
                'Prepare for possible renal replacement therapy',
                'Discuss dialysis options with nephrologist',
                'Evaluate for kidney transplant candidacy',
                'Strict dietary restrictions may apply'
            ],
            5: [
                'IMMEDIATE nephrology consultation required',
                'Dialysis evaluation and planning',
                'Kidney transplant evaluation',
                'Urgent dietary and fluid restrictions',
                'Prepare vascular access for dialysis'
            ]
        }
        
        return recommendations.get(stage, recommendations[3])
    
    def _add_footer(self, pdf: FPDF):
        """Add report footer"""
        pdf.ln(10)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(128, 128, 128)
        
        pdf.multi_cell(0, 4, 
            'This report was generated by RenalGuard AI - An AI-Assisted Kidney Health Screening Tool. '
            'This is NOT a medical diagnosis. Please consult a qualified healthcare professional '
            'for proper medical evaluation and treatment decisions.')
        
        pdf.ln(3)
        pdf.cell(0, 4, f'Report ID: {self.report_id} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C')


if __name__ == '__main__':
    # Test the report generator
    generator = ClinicalReportGenerator()
    
    test_patient = {
        'age': 55,
        'sc': 2.1,
        'bu': 55,
        'hemo': 10.5,
        'bp': 150,
        'bgr': 140,
        'sod': 138,
        'pot': 4.5,
        'pcv': 35,
        'wc': 8000,
        'rc': 4.2,
        'sg': 1.015,
        'al': 2
    }
    
    test_result = {
        'risk_level': 'HIGH',
        'risk_score': 78,
        'stage': 3,
        'egfr': 42,
        'confidence': 92
    }
    
    test_explanation = {
        'top_risk_factors': [
            {'feature': 'sc', 'feature_value': 2.1, 'contribution_pct': 35},
            {'feature': 'bu', 'feature_value': 55, 'contribution_pct': 25},
            {'feature': 'hemo', 'feature_value': 10.5, 'contribution_pct': 15},
            {'feature': 'bp', 'feature_value': 150, 'contribution_pct': 12},
            {'feature': 'age', 'feature_value': 55, 'contribution_pct': 8}
        ]
    }
    
    output_path = generator.generate_report(test_patient, test_result, test_explanation, 'test_report.pdf')
    print(f"Test report generated: {output_path}")
