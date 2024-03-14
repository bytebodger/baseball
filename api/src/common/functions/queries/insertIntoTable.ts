import { dbClient } from '../../constants/dbClient.js';
import type { Table } from '../../enums/Table.js';
import type { DatabaseValue } from '../../types/DatabaseValue.js';
import type { GenericObject } from '../../types/GenericObject.js';

export const insertIntoTable = async (table: Table, fields: GenericObject) => {
   const fieldNames: string[] = [];
   const valueParameters: string[] = [];
   const values: DatabaseValue[] = [];
   Object.entries(fields).forEach((field, index) => {
      const [fieldName, value] = field;
      fieldNames.push(fieldName);
      valueParameters.push(`$${index + 1}`);
      values.push(typeof value === 'string' ? value.trim() : value);
   })
   return await dbClient.query(
      `
         INSERT INTO
            ${table}
            (${fieldNames.join(' , ')})
         VALUES
            (${valueParameters.join(' , ')})
         RETURNING 
            *
      `,
      values,
   )
}