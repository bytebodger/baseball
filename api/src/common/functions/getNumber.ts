export const getNumber = (value?: any): number => {
   if (typeof value === 'number') return value;
   if (typeof value === 'undefined' || !value) return 0;
   if (typeof value === 'string') return Number(value);
   if (typeof value.getTime === 'function') return value.getTime();
   return Number(value);
}