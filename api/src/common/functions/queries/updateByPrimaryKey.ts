import { dbClient } from '../../constants/dbClient.js';
import { IdentifyField } from '../../enums/IdentifyField.js';
import { Table } from '../../enums/Table.js';
import { GenericObject } from '../../types/GenericObject.js';

export const updateByPrimaryKey = async (table: Table, identityField: IdentifyField, fields: GenericObject)=> {
   let set: string[] = [];
   let values: Array<string | number | boolean | null> = [];
   let valueIndex = 1;
   let identifyFieldValue = 0;
   Object.entries(fields).forEach(field => {
      const [fieldName, value] = field;
      if (fieldName === identityField)
         identifyFieldValue = value;
      if (fieldName === identityField || value === undefined)
         return;
      set.push(` ${fieldName} = $${valueIndex} `);
      values.push(typeof value === 'string' ? value.trim() : value);
      valueIndex++;
   })
   values.push(identifyFieldValue);
   const primaryKeyCondition = ` ${identityField} = $${valueIndex} `;
   return await dbClient.query(
      `
         UPDATE
            ${table}
         SET
            ${set.join(',')}
         WHERE
            ${primaryKeyCondition}
      `,
      values,
   )
}