import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any

def generate_portfolio_health_chart(applicants: List[Dict[str, Any]]):
    # Calculate Approved vs non-approved dynamically
    if not applicants:
        return go.Figure()
        
    df = pd.DataFrame(applicants)
    approved_count = len(df[df['status'] == 'Approved'])
    risk_count = len(df[df['status'] != 'Approved'])
    
    total = approved_count + risk_count
    score_text = f"{int((approved_count/total)*100)}/100" if total > 0 else "0/0"
    
    fig = go.Figure(go.Pie(
        values=[approved_count, risk_count], 
        labels=['Healthy', 'Risk'], 
        hole=.75, 
        marker_colors=['#10b981', '#1e293b'], 
        textinfo='none'
    ))
    
    fig.update_layout(
        showlegend=False, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        annotations=[dict(text=score_text, x=0.5, y=0.5, font_size=26, font_color="white", showarrow=False)]
    )
    return fig

def generate_decision_activity_chart(applicants: List[Dict[str, Any]]):
    if not applicants:
        return go.Figure()
        
    df = pd.DataFrame(applicants)
    
    # Simple grouping by status
    status_counts = df['status'].value_counts()
    
    fig = go.Figure()
    
    if 'Approved' in status_counts:
        fig.add_trace(go.Bar(x=["Total Applications"], y=[status_counts['Approved']], name='Approved', marker_color='#10b981'))
    if 'Flagged' in status_counts:
        fig.add_trace(go.Bar(x=["Total Applications"], y=[status_counts['Flagged']], name='Flagged', marker_color='#f59e0b'))
    if 'Rejected' in status_counts:
        fig.add_trace(go.Bar(x=["Total Applications"], y=[status_counts['Rejected']], name='Rejected', marker_color='#ef4444'))
        
    fig.update_layout(
        barmode='group', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#94a3b8')
    )
    return fig
