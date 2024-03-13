export const getString = (value?: any): string => {
   if (typeof value === 'string') return value;
   if (typeof value === 'undefined' || !value) return '';
   if (typeof value === 'number') return String(value);
   if (typeof value.toISOString === 'function') return value.toISOString();
   if (typeof value === 'object') return JSON.stringify(value);
   return value.toString();
}