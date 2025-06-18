import streamlit as st
import datetime

class NewsletterHTMLBuilderSES:
    def __init__(self, for_email=True):
        self.for_email = for_email
        self.base_template = """
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='UTF-8'>
            <meta name='viewport' content='width=device-width, initial-scale=1.0'>
            <title>SimPPL Newsletter - {month} {year}</title>
            {style_block}
        </head>
        <body style='{body_style}'>
        {open_tracker}
        <table align='center' width='100%' cellpadding='0' cellspacing='0'
            style='max-width:600px;margin:0 auto;padding:20px;background:#fff;
                    border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)'>
            <tr><td>
                <p style='font-size:24px;font-weight:bold;margin-bottom:8px;color:black'>
                    The <span style='background:linear-gradient(to right, #FB6A4B, #4D3EAB);
                    -webkit-background-clip:text;color:transparent'>SimPPL</span> Newsletter
                </p>
                <p style='font-size:18px;line-height:24px;margin-bottom:16px;
                        background:#000;-webkit-background-clip:text;color:transparent'>
                    {month} {year} Issue
                </p>
                {highlights_section}
                {whats_cooking_section}
                {reading_section}
                <table width='100%' cellpadding='10'
                    style='margin-top:40px;background-color:#f9f9f9;border-top:1px solid #ddd'>
                <tr>
                    <td><img src='https://simppl-newsletter-bucket.s3.ap-south-1.amazonaws.com/s3_newsletter/SimPPL-Logo-New.png'
                            alt='SimPPL Logo' style='max-width:150px;height:auto'></td>
                    <td align='right'>
                    <a href='https://twitter.com/SimPPL' target='_blank'>
                        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/X_logo.jpg/240px-X_logo.jpg'
                            style='width:30px;height:30px;margin-right:10px'></a>
                    <a href='https://linkedin.com/company/simppl' target='_blank'>
                        <img src='https://cdn-icons-png.flaticon.com/512/145/145807.png'
                            style='width:30px;height:30px'></a>
                    </td>
                </tr>
                </table>
                <div style='text-align:center;background-color:#333;color:white;
                            padding:20px 10px;border-radius:8px;margin-top:30px'>
                    <p style='margin:5px 0'>© {year} SimPPL. All rights reserved.</p>
                    <p style='margin:5px 0'>Contact us at 
                        <a href='mailto:team@simppl.org' style='color:#007BFF;text-decoration:none'>
                            team@simppl.org</a></p>
                </div>
            </td></tr>
        </table>
        </body>
        </html>
    """  

    def build_newsletter(self, content_data):
        return self.base_template.format(
            month=content_data.get("month", "Unknown"),
            year=content_data.get("year", "2025"),
            highlights_section=self._build_section(content_data.get("highlights", {}).get("items", []), "Highlights of the Month"),
            whats_cooking_section=self._build_section([content_data.get("whats_cooking", {})], "What's Cooking at SimPPL"),
            reading_section=self._build_reading_section(content_data.get("auto_links", [])),
            open_tracker="{{ses:openTracker}}" if self.for_email else "",
            style_block="" if self.for_email else self._get_preview_style(),
            body_style="font-family:Arial, sans-serif;background-color:#f9f9f9;color:#333"
                if self.for_email else
                "font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;background-color:#f9f9f9;color:#333"
        )

    def _get_preview_style(self):
        return """
            <style>
            @media (prefers-color-scheme: dark) {
                body { background-color: #1e1e1e; color: #eee; }
                a { color: #4DA8DA; }
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                background-color: #f9f9f9;
                color: #333;
            }
            </style>
            """


    def _build_section(self, items, title):
        if not items: return ""
        html = f'<p style="font-size:20px;font-weight:bold;margin:16px 0">{title}</p>'
        for item in items:
            html += f"""
            <table width='100%' style='margin-bottom:20px;border-bottom:1px solid #ddd;padding-bottom:20px'>
                <tr><td>
                    <img src='{item.get('image', '')}' alt='{item.get('title', '')}'
                         style='width:100%;height:auto;border-radius:8px;margin-bottom:20px'>
                    <p style='font-size:18px;line-height:24px;margin:16px 0;font-weight:bold'>
                        {item.get('title', 'Untitled')}</p>
                    <p style='font-size:12px;line-height:24px;margin:16px 0'>
                        {item.get('description', '')}</p>
                    <a href='{item.get('link', '#')}'
                       style='color:#007BFF;font-size:12px;text-decoration:none;display:block;text-align:right'
                       target='_blank'>Learn More</a>
                </td></tr>
            </table>
            """
        return html

    def _build_reading_section(self, links):
        if not links: return ""
        html = '<p style="font-size:20px;font-weight:bold;margin:16px 0">What We Are Reading at SimPPL</p>'
        categories = {}
        for link in links:
            category = link.get("category", "General")
            categories.setdefault(category, []).append(link)
        for category, items in categories.items():
            html += f'<p style="font-size:16px;font-weight:bold;margin:16px 0">{category}</p>'
            for item in items:
                html += f"""
                <div style='margin-bottom:20px;border:1px solid #ddd;border-radius:8px;padding:10px'>
                    <p style='font-size:14px;font-weight:bold;margin:0 0 10px'>{item.get('title', '')}</p>
                    <p style='font-size:12px;margin:0 0 10px'>{item.get('summary', '')}</p>
                    <a href='{item.get('url', '#')}'
                       style='color:#007BFF;font-size:12px;text-decoration:none'
                       target='_blank'>Learn More</a>
                </div>
                """
        return html

def create_newsletter_builder():
    st.title("📰 SimPPL Newsletter Builder")

    if 'highlights' not in st.session_state:
        st.session_state.highlights = [{'title': '', 'description': '', 'link': '', 'image': ''} for _ in range(3)]

    if 'cooking' not in st.session_state:
        st.session_state.cooking = {'title': '', 'description': '', 'link': '', 'image': ''}

    if 'newsletter_data' not in st.session_state:
        st.session_state.newsletter_data = {}

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🌟 Highlights of the Month")
        for i, highlight in enumerate(st.session_state.highlights):
            with st.expander(f"Highlight {i + 1}", expanded=True):
                highlight['title'] = st.text_input(f"Title {i + 1}", value=highlight['title'], key=f"title_{i}")
                highlight['description'] = st.text_area(f"Description {i + 1}", value=highlight['description'], key=f"desc_{i}")
                highlight['link'] = st.text_input(f"Link {i + 1}", value=highlight['link'], key=f"link_{i}")
                highlight['image'] = st.text_input(f"Image URL {i + 1}", value=highlight['image'], key=f"img_{i}")

        st.subheader("👨‍🍳 What's Cooking")
        cook = st.session_state.cooking
        cook['title'] = st.text_input("Project Title", value=cook['title'], key="cook_title")
        cook['description'] = st.text_area("Project Description", value=cook['description'], key="cook_desc")
        cook['link'] = st.text_input("Project Link", value=cook['link'], key="cook_link")
        cook['image'] = st.text_input("Project Image URL", value=cook['image'], key="cook_img")

    with col2:
        st.subheader("🔁 Auto-Curated Links")

        def get_automated_links(month, year):
            return [
                {
                    'id': 'link1',
                    'title': 'LLMs in Production',
                    'url': 'https://example.com/llms',
                    'summary': 'How large language models are used in real-world apps.',
                    'category': 'Technical'
                },
                {
                    'id': 'link2',
                    'title': 'Responsible AI Governance',
                    'url': 'https://example.com/policy',
                    'summary': 'Principles and challenges in policy frameworks for AI.',
                    'category': 'Policy'
                }
            ]

        selected_month = st.session_state.get('selected_month', datetime.date.today().month)
        selected_year = st.session_state.get('selected_year', datetime.date.today().year)
        auto_links = get_automated_links(selected_month, selected_year)

        if st.button("🚀 Generate Newsletter"):
            content_data = {
                'month': datetime.date(2000, selected_month, 1).strftime('%B'),
                'year': selected_year,
                'highlights': {'items': [h for h in st.session_state.highlights if h['title'].strip()]},
                'whats_cooking': st.session_state.cooking,
                'auto_links': auto_links
            }
            st.session_state.newsletter_data = content_data
            st.success("✅ Newsletter generated! Scroll down to preview →")

    if st.session_state.newsletter_data:
        st.divider()
        st.header("👀 Preview & Export")
        toggle_email = st.toggle("📤 View Email-Safe Version (Inline CSS)", value=True)
        preview_width = st.selectbox("📱 Preview Width", ["375px (Mobile)", "768px (Tablet)", "100% (Desktop)"])
        width_css = preview_width.split()[0]
        height = 700 if "Mobile" in width_css else 850

        builder = NewsletterHTMLBuilderSES(for_email=toggle_email)
        html = builder.build_newsletter(st.session_state.newsletter_data)

        st.components.v1.html(f"<div style='width:{width_css};margin:auto'>{html}</div>", height=height, scrolling=True)
        st.download_button("💾 Download HTML", data=html, file_name="newsletter.html", mime="text/html")
        with st.expander("🧾 View HTML Source"):
            st.code(html, language="html")
