import { dbClient } from '../../constants/dbClient.js';
import type { IdentityField } from '../../enums/IdentityField.js';
import type { Table } from '../../enums/Table.js';
import type { DatabaseValue } from '../../types/DatabaseValue.js';
import type { GenericObject } from '../../types/GenericObject.js';

export const updateDBTableByPrimaryKey = async (table: Table, identityField: IdentityField, fields: GenericObject) => {
   const set: string[] = [];
   const values: DatabaseValue[] = [];
   let valueIndex = 1;
   let identityFieldValue = 0;
   Object.entries(fields).forEach(field => {
      const [fieldName, value] = field;
      if (fieldName === identityField)
         identityFieldValue = value;
      if (fieldName === identityField || value === undefined)
         return;
      set.push(` ${fieldName} = $${valueIndex} `);
      values.push(typeof value === 'string' ? value.trim() : value);
      valueIndex++;
   })
   values.push(identityFieldValue);
   const primaryKeyCondition = ` ${identityField} = $${valueIndex} `;
   return await dbClient.query(
      `
         UPDATE
            ${table}
         SET
            ${set.join(' , ')}
         WHERE
            ${primaryKeyCondition}
      `,
      values,
   )
}