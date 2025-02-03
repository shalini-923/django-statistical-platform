from django import template
import base64

register = template.Library()

@register.filter(name='base64_encode')
def base64_encode(value):
    if isinstance(value, bytes):  # Ensure input is in bytes
        return base64.b64encode(value).decode('utf-8')  # Return the base64 string
    return value  # Return original value if it's not bytes
